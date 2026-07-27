use crate::mobjects::mesh_2d::{GeometryFingerprint, MeshGeometry2D, TriangleMesh2D, Vertex2D};
use crate::mobjects::mesh_3d::{TriangleMesh3D, Vertex};
use crate::video_backend::vulkan_h264::VulkanVideoFrame;
use crate::vulkan::context::{TimelineSemaphore, VulkanContext};
use ash::vk;
use ash::vk::Handle;
use ash::vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_MAIN;
use std::collections::HashMap;
use std::sync::Arc;

// The encoder keeps 8 frames in flight; one extra image prevents the renderer
// from overwriting an image before submit-side backpressure can release a slot.
const VIDEO_NV12_IMAGE_COUNT: usize = 9;
const VK_FORMAT_G8_B8R8_2PLANE_420_UNORM_RAW: i32 = 1_000_156_003;
const VK_IMAGE_USAGE_VIDEO_ENCODE_SRC_BIT_KHR_RAW: u32 = 0x0000_4000;
const VK_IMAGE_ASPECT_PLANE_0_BIT_RAW: u32 = 0x0000_0010;
const VK_IMAGE_ASPECT_PLANE_1_BIT_RAW: u32 = 0x0000_0020;
const RENDER_FRAME_COUNT: usize = 3;
const GPU_TIMESTAMP_COUNT: u32 = 6;

fn align_up(value: u64, alignment: u64) -> u64 {
    if alignment <= 1 {
        value
    } else {
        (value + alignment - 1) & !(alignment - 1)
    }
}

fn video_timeline_values(next_ready_value: u64) -> (Option<u64>, u64, u64) {
    assert!(
        next_ready_value % 2 == 1,
        "video ready timeline values must be odd"
    );
    (
        (next_ready_value > 1).then_some(next_ready_value - 1),
        next_ready_value,
        next_ready_value + 1,
    )
}

fn compile_wgsl_full(ctx: &VulkanContext, source: &str) -> vk::ShaderModule {
    let module = naga::front::wgsl::parse_str(source).unwrap();
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    let info = validator.validate(&module).unwrap();
    let options = naga::back::spv::Options::default();
    let spv = naga::back::spv::write_vec(&module, &info, &options, None).unwrap();
    let create_info = vk::ShaderModuleCreateInfo {
        s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO,
        code_size: spv.len() * 4,
        p_code: spv.as_ptr(),
        ..Default::default()
    };
    unsafe { ctx.device.create_shader_module(&create_info, None).unwrap() }
}

pub struct Buffer {
    pub vk_buffer: vk::Buffer,
    pub allocation: Option<gpu_allocator::vulkan::Allocation>,
    pub size: u64,
}

impl Buffer {
    pub fn new(
        ctx: &VulkanContext,
        size: u64,
        usage: vk::BufferUsageFlags,
        memory_location: gpu_allocator::MemoryLocation,
    ) -> Self {
        let buffer_info = vk::BufferCreateInfo {
            s_type: vk::StructureType::BUFFER_CREATE_INFO,
            size,
            usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        let vk_buffer = unsafe { ctx.device.create_buffer(&buffer_info, None).unwrap() };
        let requirements = unsafe { ctx.device.get_buffer_memory_requirements(vk_buffer) };

        let allocation = ctx
            .allocator
            .lock()
            .unwrap()
            .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                name: "buffer",
                requirements,
                location: memory_location,
                linear: true,
                allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            })
            .unwrap();

        unsafe {
            ctx.device
                .bind_buffer_memory(vk_buffer, allocation.memory(), allocation.offset())
                .unwrap();
        }

        Self {
            vk_buffer,
            allocation: Some(allocation),
            size,
        }
    }

    pub fn write_bytes(&self, offset: u64, data: &[u8]) {
        if let Some(alloc) = &self.allocation {
            if let Some(mapped) = alloc.mapped_ptr() {
                unsafe {
                    let dst = mapped.as_ptr().add(offset as usize);
                    std::ptr::copy_nonoverlapping(data.as_ptr(), dst as *mut u8, data.len());
                }
            }
        }
    }

    pub fn destroy(&mut self, ctx: &VulkanContext) {
        unsafe {
            ctx.device.destroy_buffer(self.vk_buffer, None);
        }
        if let Some(allocation) = self.allocation.take() {
            ctx.allocator.lock().unwrap().free(allocation).unwrap();
        }
    }
}

fn msaa_to_vk_sample_count(msaa_samples: u32) -> vk::SampleCountFlags {
    match msaa_samples {
        1 => vk::SampleCountFlags::TYPE_1,
        2 => vk::SampleCountFlags::TYPE_2,
        4 => vk::SampleCountFlags::TYPE_4,
        8 => vk::SampleCountFlags::TYPE_8,
        16 => vk::SampleCountFlags::TYPE_16,
        32 => vk::SampleCountFlags::TYPE_32,
        64 => vk::SampleCountFlags::TYPE_64,
        _ => vk::SampleCountFlags::TYPE_8,
    }
}

fn get_max_usable_sample_count(
    ctx: &VulkanContext,
    requested: vk::SampleCountFlags,
) -> vk::SampleCountFlags {
    let properties = unsafe {
        ctx.instance
            .get_physical_device_properties(ctx.physical_device)
    };
    let counts = properties.limits.framebuffer_color_sample_counts
        & properties.limits.framebuffer_depth_sample_counts;

    if counts.contains(requested) {
        return requested;
    }

    // Fallback to highest supported
    let fallbacks = [
        vk::SampleCountFlags::TYPE_64,
        vk::SampleCountFlags::TYPE_32,
        vk::SampleCountFlags::TYPE_16,
        vk::SampleCountFlags::TYPE_8,
        vk::SampleCountFlags::TYPE_4,
        vk::SampleCountFlags::TYPE_2,
        vk::SampleCountFlags::TYPE_1,
    ];

    for &fallback in &fallbacks {
        if counts.contains(fallback) && fallback.as_raw() <= requested.as_raw() {
            return fallback;
        }
    }
    vk::SampleCountFlags::TYPE_1
}

pub struct Image {
    pub vk_image: vk::Image,
    pub allocation: Option<gpu_allocator::vulkan::Allocation>,
    pub view: vk::ImageView,
    pub format: vk::Format,
    pub width: u32,
    pub height: u32,
}

impl Image {
    pub fn new(
        ctx: &VulkanContext,
        width: u32,
        height: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        aspect_mask: vk::ImageAspectFlags,
        samples: vk::SampleCountFlags,
    ) -> Self {
        let image_info = vk::ImageCreateInfo {
            s_type: vk::StructureType::IMAGE_CREATE_INFO,
            image_type: vk::ImageType::TYPE_2D,
            format,
            extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            samples,
            tiling: vk::ImageTiling::OPTIMAL,
            usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            ..Default::default()
        };

        let vk_image = unsafe { ctx.device.create_image(&image_info, None).unwrap() };
        let requirements = unsafe { ctx.device.get_image_memory_requirements(vk_image) };

        let allocation = ctx
            .allocator
            .lock()
            .unwrap()
            .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                name: "image",
                requirements,
                location: gpu_allocator::MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            })
            .unwrap();

        unsafe {
            ctx.device
                .bind_image_memory(vk_image, allocation.memory(), allocation.offset())
                .unwrap();
        }

        let view_info = vk::ImageViewCreateInfo {
            s_type: vk::StructureType::IMAGE_VIEW_CREATE_INFO,
            image: vk_image,
            view_type: vk::ImageViewType::TYPE_2D,
            format,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
            ..Default::default()
        };

        let view = unsafe { ctx.device.create_image_view(&view_info, None).unwrap() };

        Self {
            vk_image,
            allocation: Some(allocation),
            view,
            format,
            width,
            height,
        }
    }

    pub fn destroy(&mut self, ctx: &VulkanContext) {
        unsafe {
            ctx.device.destroy_image_view(self.view, None);
            ctx.device.destroy_image(self.vk_image, None);
        }
        if let Some(allocation) = self.allocation.take() {
            ctx.allocator.lock().unwrap().free(allocation).unwrap();
        }
    }
}

pub struct VideoNv12Image {
    pub vk_image: vk::Image,
    pub allocation: Option<gpu_allocator::vulkan::Allocation>,
    pub color_view: vk::ImageView,
    pub y_view: vk::ImageView,
    pub uv_view: vk::ImageView,
    pub format: vk::Format,
    pub width: u32,
    pub height: u32,
}

impl VideoNv12Image {
    pub fn new(ctx: &VulkanContext, width: u32, height: u32) -> Self {
        let format = vk::Format::from_raw(VK_FORMAT_G8_B8R8_2PLANE_420_UNORM_RAW);
        let usage = vk::ImageUsageFlags::STORAGE
            | vk::ImageUsageFlags::from_raw(VK_IMAGE_USAGE_VIDEO_ENCODE_SRC_BIT_KHR_RAW);
        let mut queue_family_indices = Vec::new();
        queue_family_indices.push(ctx.queue_family_index);
        if let Some(video_queue_family_index) = ctx.video_encode_queue_family_index {
            if video_queue_family_index != ctx.queue_family_index {
                queue_family_indices.push(video_queue_family_index);
            }
        }
        let sharing_mode = if queue_family_indices.len() > 1 {
            vk::SharingMode::CONCURRENT
        } else {
            vk::SharingMode::EXCLUSIVE
        };
        let mut h264_profile = vk::VideoEncodeH264ProfileInfoKHR::default()
            .std_profile_idc(StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_MAIN);
        let profile = vk::VideoProfileInfoKHR::default()
            .video_codec_operation(vk::VideoCodecOperationFlagsKHR::ENCODE_H264)
            .chroma_subsampling(vk::VideoChromaSubsamplingFlagsKHR::TYPE_420)
            .luma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
            .chroma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
            .push_next(&mut h264_profile);
        let profiles = [profile];
        let mut profile_list = vk::VideoProfileListInfoKHR::default().profiles(&profiles);
        let image_info = vk::ImageCreateInfo {
            s_type: vk::StructureType::IMAGE_CREATE_INFO,
            p_next: (&mut profile_list as *mut vk::VideoProfileListInfoKHR).cast(),
            flags: vk::ImageCreateFlags::MUTABLE_FORMAT | vk::ImageCreateFlags::EXTENDED_USAGE,
            image_type: vk::ImageType::TYPE_2D,
            format,
            extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage,
            sharing_mode,
            queue_family_index_count: queue_family_indices.len() as u32,
            p_queue_family_indices: queue_family_indices.as_ptr(),
            initial_layout: vk::ImageLayout::UNDEFINED,
            ..Default::default()
        };

        let vk_image = unsafe { ctx.device.create_image(&image_info, None).unwrap() };
        let requirements = unsafe { ctx.device.get_image_memory_requirements(vk_image) };
        let allocation = ctx
            .allocator
            .lock()
            .unwrap()
            .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                name: "video-nv12-image",
                requirements,
                location: gpu_allocator::MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            })
            .unwrap();

        unsafe {
            ctx.device
                .bind_image_memory(vk_image, allocation.memory(), allocation.offset())
                .unwrap();
        }

        let color_view = Self::create_view(
            ctx,
            vk_image,
            format,
            vk::ImageAspectFlags::COLOR,
            vk::ImageUsageFlags::from_raw(VK_IMAGE_USAGE_VIDEO_ENCODE_SRC_BIT_KHR_RAW),
        );
        let y_view = Self::create_view(
            ctx,
            vk_image,
            vk::Format::R8_UNORM,
            vk::ImageAspectFlags::from_raw(VK_IMAGE_ASPECT_PLANE_0_BIT_RAW),
            vk::ImageUsageFlags::STORAGE,
        );
        let uv_view = Self::create_view(
            ctx,
            vk_image,
            vk::Format::R8G8_UNORM,
            vk::ImageAspectFlags::from_raw(VK_IMAGE_ASPECT_PLANE_1_BIT_RAW),
            vk::ImageUsageFlags::STORAGE,
        );

        Self {
            vk_image,
            allocation: Some(allocation),
            color_view,
            y_view,
            uv_view,
            format,
            width,
            height,
        }
    }

    fn create_view(
        ctx: &VulkanContext,
        image: vk::Image,
        format: vk::Format,
        aspect_mask: vk::ImageAspectFlags,
        usage: vk::ImageUsageFlags,
    ) -> vk::ImageView {
        let mut usage_info = vk::ImageViewUsageCreateInfo::default().usage(usage);
        let view_info = vk::ImageViewCreateInfo {
            s_type: vk::StructureType::IMAGE_VIEW_CREATE_INFO,
            p_next: (&mut usage_info as *mut vk::ImageViewUsageCreateInfo).cast(),
            image,
            view_type: vk::ImageViewType::TYPE_2D,
            format,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
            ..Default::default()
        };
        unsafe { ctx.device.create_image_view(&view_info, None).unwrap() }
    }

    pub fn destroy(&mut self, ctx: &VulkanContext) {
        unsafe {
            ctx.device.destroy_image_view(self.uv_view, None);
            ctx.device.destroy_image_view(self.y_view, None);
            ctx.device.destroy_image_view(self.color_view, None);
            ctx.device.destroy_image(self.vk_image, None);
        }
        if let Some(allocation) = self.allocation.take() {
            ctx.allocator.lock().unwrap().free(allocation).unwrap();
        }
    }
}

pub struct VideoNv12Slot {
    pub image: VideoNv12Image,
    pub descriptor_set: vk::DescriptorSet,
    pub layout: vk::ImageLayout,
    pub timeline: Arc<TimelineSemaphore>,
    pub next_ready_value: u64,
    pub last_ready_value: Option<u64>,
    pub frame_available: bool,
}

impl VideoNv12Slot {
    fn new(ctx: &Arc<VulkanContext>, width: u32, height: u32) -> Self {
        Self {
            image: VideoNv12Image::new(ctx, width, height),
            descriptor_set: vk::DescriptorSet::null(),
            layout: vk::ImageLayout::UNDEFINED,
            timeline: TimelineSemaphore::new(ctx, 0)
                .expect("failed to create video frame timeline semaphore"),
            next_ready_value: 1,
            last_ready_value: None,
            frame_available: false,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Nv12Constants {
    pub width: u32,
    pub height: u32,
    pub _padding: [u32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform2D {
    pub width: f32,
    pub height: f32,
    pub scale_factor: f32,
    pub _pad: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Instance2D {
    pub model_0: [f32; 4],
    pub model_1: [f32; 4],
    pub model_2: [f32; 4],
    pub model_3: [f32; 4],
    pub color: [f32; 4],
}

impl Instance2D {
    fn new(transform: nalgebra::Matrix4<crate::GMFloat>, color: [f32; 4]) -> Self {
        Self {
            model_0: [
                transform[(0, 0)] as f32,
                transform[(1, 0)] as f32,
                transform[(2, 0)] as f32,
                transform[(3, 0)] as f32,
            ],
            model_1: [
                transform[(0, 1)] as f32,
                transform[(1, 1)] as f32,
                transform[(2, 1)] as f32,
                transform[(3, 1)] as f32,
            ],
            model_2: [
                transform[(0, 2)] as f32,
                transform[(1, 2)] as f32,
                transform[(2, 2)] as f32,
                transform[(3, 2)] as f32,
            ],
            model_3: [
                transform[(0, 3)] as f32,
                transform[(1, 3)] as f32,
                transform[(2, 3)] as f32,
                transform[(3, 3)] as f32,
            ],
            color,
        }
    }
}

struct Mesh2DSubmission {
    geometry: Arc<MeshGeometry2D>,
    instance: Instance2D,
}

struct Mesh2DBatch {
    geometry: Arc<MeshGeometry2D>,
    instances: Vec<Instance2D>,
}

#[derive(Clone)]
struct CachedMesh2D {
    geometry: Arc<MeshGeometry2D>,
    vertex_offset: u64,
    index_offset: u64,
    index_count: u32,
}

struct GeometryUpload2D {
    geometry: Arc<MeshGeometry2D>,
    staging_vertex_offset: u64,
    staging_index_offset: u64,
    device_vertex_offset: u64,
    device_index_offset: u64,
}

struct PreparedMesh2DBatch {
    first_index: u32,
    vertex_offset: i32,
    index_count: u32,
    first_instance: u32,
    instance_count: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PrepareMesh2DError {
    StaticArenaExhausted,
    FrameStagingArenaExhausted,
    FrameInstanceArenaExhausted,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RendererStats {
    pub mesh_2d_draw_calls: u32,
    pub mesh_2d_instances: u32,
    pub mesh_2d_geometry_uploads: u32,
    pub mesh_2d_vertex_bytes_uploaded: u64,
    pub mesh_2d_index_bytes_uploaded: u64,
    pub mesh_2d_arena_rebuilds: u32,
    pub sdf_dispatches: u32,
    pub raster_passes: u32,
    pub depth_attachment_raster_passes: u32,
    pub downsample_dispatches: u32,
    pub fused_video_downsample_dispatches: u32,
    pub composite_dispatches: u32,
    pub output_conversion_dispatches: u32,
    pub rgba_readback_copies: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GpuPassTimings {
    pub frame_ms: f64,
    pub geometry_upload_ms: f64,
    pub sdf_ms: f64,
    pub raster_ms: f64,
    pub postprocess_ms: f64,
    pub output_ms: f64,
}

impl GpuPassTimings {
    fn from_timestamps(
        timestamps: [u64; GPU_TIMESTAMP_COUNT as usize],
        timestamp_period_ns: f32,
        timestamp_valid_bits: u32,
        plan: FrameExecutionPlan,
        has_geometry_upload: bool,
        has_postprocess: bool,
        has_output: bool,
    ) -> Self {
        let elapsed_ms = |start: usize, end: usize| {
            timestamp_delta(timestamps[start], timestamps[end], timestamp_valid_bits) as f64
                * timestamp_period_ns as f64
                / 1_000_000.0
        };
        Self {
            frame_ms: elapsed_ms(0, 5),
            geometry_upload_ms: has_geometry_upload
                .then(|| elapsed_ms(0, 1))
                .unwrap_or_default(),
            sdf_ms: plan
                .runs_sdf()
                .then(|| elapsed_ms(1, 2))
                .unwrap_or_default(),
            raster_ms: plan
                .runs_raster()
                .then(|| elapsed_ms(2, 3))
                .unwrap_or_default(),
            postprocess_ms: has_postprocess
                .then(|| elapsed_ms(3, 4))
                .unwrap_or_default(),
            output_ms: has_output.then(|| elapsed_ms(4, 5)).unwrap_or_default(),
        }
    }
}

fn timestamp_delta(start: u64, end: u64, valid_bits: u32) -> u64 {
    let mask = if valid_bits >= 64 {
        u64::MAX
    } else {
        (1u64 << valid_bits) - 1
    };
    end.wrapping_sub(start) & mask
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FrameExecutionPlan {
    Empty,
    SdfOnly,
    RasterDirect,
    RasterDownsample,
    SdfRasterComposite,
}

impl FrameExecutionPlan {
    fn build(has_sdf: bool, has_raster: bool, ssaa_factor: u32) -> Self {
        match (has_sdf, has_raster, ssaa_factor) {
            (false, false, _) => Self::Empty,
            (true, false, _) => Self::SdfOnly,
            (false, true, 1) => Self::RasterDirect,
            (false, true, _) => Self::RasterDownsample,
            (true, true, _) => Self::SdfRasterComposite,
        }
    }

    fn runs_sdf(self) -> bool {
        matches!(self, Self::SdfOnly | Self::SdfRasterComposite)
    }

    fn runs_raster(self) -> bool {
        matches!(
            self,
            Self::RasterDirect | Self::RasterDownsample | Self::SdfRasterComposite
        )
    }
}

#[derive(Clone, Copy, Debug)]
struct TrackedImageState {
    layout: vk::ImageLayout,
    stage: vk::PipelineStageFlags2,
    access: vk::AccessFlags2,
}

impl TrackedImageState {
    const UNDEFINED: Self = Self {
        layout: vk::ImageLayout::UNDEFINED,
        stage: vk::PipelineStageFlags2::NONE,
        access: vk::AccessFlags2::NONE,
    };
}

unsafe fn transition_image(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    image: vk::Image,
    aspect_mask: vk::ImageAspectFlags,
    state: &mut TrackedImageState,
    next: TrackedImageState,
) {
    let barrier = vk::ImageMemoryBarrier2::default()
        .src_stage_mask(state.stage)
        .src_access_mask(state.access)
        .dst_stage_mask(next.stage)
        .dst_access_mask(next.access)
        .old_layout(state.layout)
        .new_layout(next.layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });
    let dependency =
        vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&barrier));
    unsafe {
        device.cmd_pipeline_barrier2(command_buffer, &dependency);
    }
    *state = next;
}

unsafe fn write_gpu_timestamp(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    query_pool: vk::QueryPool,
    query: u32,
    enabled: bool,
) {
    if enabled {
        unsafe {
            device.cmd_write_timestamp2(
                command_buffer,
                vk::PipelineStageFlags2::ALL_COMMANDS,
                query_pool,
                query,
            );
        }
    }
}

fn build_ordered_mesh_2d_batches(submissions: Vec<Mesh2DSubmission>) -> Vec<Mesh2DBatch> {
    let mut batches: Vec<Mesh2DBatch> = Vec::new();
    for submission in submissions {
        if let Some(last) = batches.last_mut() {
            if last.geometry.same_geometry(&submission.geometry) {
                last.instances.push(submission.instance);
                continue;
            }
        }
        batches.push(Mesh2DBatch {
            geometry: submission.geometry,
            instances: vec![submission.instance],
        });
    }
    batches
}

fn prepare_mesh_2d_batches(
    mesh_cache: &mut HashMap<GeometryFingerprint, Vec<CachedMesh2D>>,
    static_vertex_used: &mut u64,
    static_index_used: &mut u64,
    static_vertex_capacity: u64,
    static_index_capacity: u64,
    staging_vertex_capacity: u64,
    staging_index_capacity: u64,
    instance_capacity: u64,
    batches: &[Mesh2DBatch],
) -> Result<
    (
        Vec<PreparedMesh2DBatch>,
        Vec<GeometryUpload2D>,
        Vec<Instance2D>,
    ),
    PrepareMesh2DError,
> {
    let mut prepared = Vec::with_capacity(batches.len());
    let mut uploads = Vec::new();
    let mut instances = Vec::new();
    let mut staging_vertex_used = 0u64;
    let mut staging_index_used = 0u64;

    for batch in batches {
        if batch.geometry.indices().is_empty() || batch.instances.is_empty() {
            continue;
        }
        let fingerprint = batch.geometry.fingerprint();
        let cached = mesh_cache.get(&fingerprint).and_then(|entries| {
            entries
                .iter()
                .find(|entry| entry.geometry.same_geometry(&batch.geometry))
                .cloned()
        });
        let cached = match cached {
            Some(cached) => cached,
            None => {
                let vertex_size = std::mem::size_of_val(batch.geometry.vertices()) as u64;
                let index_size = std::mem::size_of_val(batch.geometry.indices()) as u64;
                let device_vertex_offset = align_up(*static_vertex_used, 4);
                let device_index_offset = align_up(*static_index_used, 4);
                let staging_vertex_offset = align_up(staging_vertex_used, 4);
                let staging_index_offset = align_up(staging_index_used, 4);

                if device_vertex_offset + vertex_size > static_vertex_capacity
                    || device_index_offset + index_size > static_index_capacity
                {
                    return Err(PrepareMesh2DError::StaticArenaExhausted);
                }
                if staging_vertex_offset + vertex_size > staging_vertex_capacity
                    || staging_index_offset + index_size > staging_index_capacity
                {
                    return Err(PrepareMesh2DError::FrameStagingArenaExhausted);
                }

                let cached = CachedMesh2D {
                    geometry: batch.geometry.clone(),
                    vertex_offset: device_vertex_offset,
                    index_offset: device_index_offset,
                    index_count: batch.geometry.indices().len() as u32,
                };
                mesh_cache
                    .entry(fingerprint)
                    .or_default()
                    .push(cached.clone());
                uploads.push(GeometryUpload2D {
                    geometry: batch.geometry.clone(),
                    staging_vertex_offset,
                    staging_index_offset,
                    device_vertex_offset,
                    device_index_offset,
                });
                *static_vertex_used = device_vertex_offset + vertex_size;
                *static_index_used = device_index_offset + index_size;
                staging_vertex_used = staging_vertex_offset + vertex_size;
                staging_index_used = staging_index_offset + index_size;
                cached
            }
        };

        let first_instance = instances.len() as u32;
        instances.extend_from_slice(&batch.instances);
        prepared.push(PreparedMesh2DBatch {
            first_index: (cached.index_offset / std::mem::size_of::<u32>() as u64) as u32,
            vertex_offset: (cached.vertex_offset / std::mem::size_of::<Vertex2D>() as u64) as i32,
            index_count: cached.index_count,
            first_instance,
            instance_count: batch.instances.len() as u32,
        });
    }

    if std::mem::size_of_val(instances.as_slice()) as u64 > instance_capacity {
        return Err(PrepareMesh2DError::FrameInstanceArenaExhausted);
    }
    Ok((prepared, uploads, instances))
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub pos: [f32; 3],
    pub _padding0: u32,
    pub look_at: [f32; 3],
    pub _padding1: u32,
    pub up: [f32; 3],
    pub fov: f32,
    pub width: f32,
    pub height: f32,
    pub proj_type: u32,
    pub ortho_left: f32,
    pub ortho_right: f32,
    pub ortho_bottom: f32,
    pub ortho_top: f32,
    pub has_clip: u32,
    pub clip_x: f32,
    pub clip_y: f32,
    pub clip_w: f32,
    pub clip_h: f32,
    pub aa_level: u32,
    pub num_primitives: u32,
    pub _pad2: u32,
    pub _pad3: u32,
    pub proj_mat: [f32; 16],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PrimitiveData3D {
    pub color: [f32; 4],
    pub params: [f32; 12],
    pub shape_type: u32,
    pub padding: [u32; 3],
}

struct RenderTargetSet {
    texture: Image,
    texture_state: TrackedImageState,
    resolved_texture: Image,
    resolved_texture_state: TrackedImageState,
    compute_descriptor_set: vk::DescriptorSet,
    composite_descriptor_set: vk::DescriptorSet,
}

impl RenderTargetSet {
    fn destroy(&mut self, ctx: &VulkanContext) {
        self.texture.destroy(ctx);
        self.resolved_texture.destroy(ctx);
    }
}

pub struct RenderCache {
    pub width: u32,
    pub height: u32,
    render_targets: [RenderTargetSet; RENDER_FRAME_COUNT],
    pub msaa_texture: Option<Image>,
    msaa_texture_state: TrackedImageState,
    pub msaa_depth_texture: Option<Image>,
    msaa_depth_texture_state: TrackedImageState,
    pub output_buffers: [Buffer; RENDER_FRAME_COUNT],
    pub nv12_output_buffers: [Buffer; RENDER_FRAME_COUNT],
    pub nv12_descriptor_sets: [vk::DescriptorSet; RENDER_FRAME_COUNT],
    pub yuv444p_output_buffers: [Buffer; RENDER_FRAME_COUNT],
    pub yuv444p_descriptor_sets: [vk::DescriptorSet; RENDER_FRAME_COUNT],
    pub video_nv12_slots: Vec<VideoNv12Slot>,
    pub current_frame: usize,
    pub raster_descriptor_set: vk::DescriptorSet,
    pub raster_descriptor_set_2d: vk::DescriptorSet,
    pub padded_bytes_per_row: u32,
    pub rgba_preview_buffer: Vec<u8>,
}

impl RenderCache {
    pub fn destroy(&mut self, ctx: &VulkanContext) {
        for targets in &mut self.render_targets {
            targets.destroy(ctx);
        }
        if let Some(texture) = &mut self.msaa_texture {
            texture.destroy(ctx);
        }
        if let Some(texture) = &mut self.msaa_depth_texture {
            texture.destroy(ctx);
        }
        for buf in &mut self.output_buffers {
            buf.destroy(ctx);
        }
        for buf in &mut self.nv12_output_buffers {
            buf.destroy(ctx);
        }
        for buf in &mut self.yuv444p_output_buffers {
            buf.destroy(ctx);
        }
        for slot in &mut self.video_nv12_slots {
            slot.image.destroy(ctx);
        }
    }
}

pub struct FrameData {
    pub command_pool: vk::CommandPool,
    pub command_buffer: vk::CommandBuffer,
    pub fence: vk::Fence,
    query_pool: vk::QueryPool,
    timestamps_pending: bool,
    profiled_plan: FrameExecutionPlan,
    profiled_geometry_upload: bool,
    profiled_postprocess: bool,
    profiled_output: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct RenderOutputs {
    pub cpu_nv12: bool,
    pub vulkan_video: bool,
    pub cpu_rgba: bool,
    pub cpu_yuv444p: bool,
}

impl RenderOutputs {
    pub const ALL: Self = Self {
        cpu_nv12: true,
        vulkan_video: true,
        cpu_rgba: true,
        cpu_yuv444p: true,
    };

    pub const VULKAN_VIDEO_ONLY: Self = Self {
        cpu_nv12: false,
        vulkan_video: true,
        cpu_rgba: false,
        cpu_yuv444p: false,
    };

    pub const CPU_NV12_ONLY: Self = Self {
        cpu_nv12: true,
        vulkan_video: false,
        cpu_rgba: false,
        cpu_yuv444p: false,
    };

    pub const CPU_RGBA_ONLY: Self = Self {
        cpu_nv12: false,
        vulkan_video: false,
        cpu_rgba: true,
        cpu_yuv444p: false,
    };

    pub const CPU_READBACKS: Self = Self {
        cpu_nv12: true,
        vulkan_video: false,
        cpu_rgba: true,
        cpu_yuv444p: true,
    };
}

pub struct VulkanRenderer {
    ctx: Arc<VulkanContext>,
    msaa_samples: u32,
    ssaa_factor: u32,

    descriptor_pool: vk::DescriptorPool,
    compute_descriptor_set_layout: vk::DescriptorSetLayout,
    raster_descriptor_set_layout: vk::DescriptorSetLayout,
    raster_descriptor_set_layout_2d: vk::DescriptorSetLayout,
    composite_descriptor_set_layout: vk::DescriptorSetLayout,
    nv12_descriptor_set_layout: vk::DescriptorSetLayout,
    video_nv12_descriptor_set_layout: vk::DescriptorSetLayout,

    compute_pipeline_layout: vk::PipelineLayout,
    compute_pipeline: vk::Pipeline,
    composite_pipeline_layout: vk::PipelineLayout,
    composite_pipeline: vk::Pipeline,
    downsample_pipeline: vk::Pipeline,
    nv12_pipeline_layout: vk::PipelineLayout,
    nv12_pipeline: vk::Pipeline,
    video_nv12_pipeline_layout: vk::PipelineLayout,
    video_nv12_pipeline: vk::Pipeline,
    video_nv12_downsample_pipeline: vk::Pipeline,
    yuv444p_pipeline: vk::Pipeline,

    raster_pipeline_layout: vk::PipelineLayout,
    raster_pipeline: vk::Pipeline,

    vertex_buffer: Buffer,
    index_buffer: Buffer,
    camera_buffer: Buffer,
    buffer_3d: Buffer,
    nv12_constants_buffer: Buffer,

    raster_pipeline_layout_2d: vk::PipelineLayout,
    raster_pipeline_2d: vk::Pipeline,
    raster_pipeline_2d_depthless: vk::Pipeline,
    vertex_buffer_2d: Buffer,
    index_buffer_2d: Buffer,
    vertex_staging_buffer_2d: Buffer,
    index_staging_buffer_2d: Buffer,
    instance_buffer_2d: Buffer,
    camera_buffer_2d: Buffer,
    vertex_buffer_stride: u64,
    index_buffer_stride: u64,
    camera_buffer_stride: u64,
    primitive_buffer_stride: u64,
    vertex_staging_buffer_2d_stride: u64,
    index_staging_buffer_2d_stride: u64,
    instance_buffer_2d_stride: u64,
    camera_buffer_2d_stride: u64,
    mesh_cache_2d: HashMap<GeometryFingerprint, Vec<CachedMesh2D>>,
    static_vertex_buffer_2d_used: u64,
    static_index_buffer_2d_used: u64,
    last_stats: RendererStats,
    gpu_profiling: bool,
    last_gpu_timings: Option<GpuPassTimings>,

    frame_data: [FrameData; RENDER_FRAME_COUNT],

    cache: std::sync::Mutex<Option<RenderCache>>,
}

impl VulkanRenderer {
    pub fn new(ctx: Arc<VulkanContext>, config: crate::RendererConfig) -> Self {
        let msaa_samples = config.msaa_samples;
        let ssaa_factor = config.ssaa_factor;
        let requested_sample_count = msaa_to_vk_sample_count(msaa_samples);
        let sample_count = get_max_usable_sample_count(&ctx, requested_sample_count);

        let compute_shader = compile_wgsl_full(&ctx, include_str!("shader.wgsl"));
        let raster_shader = compile_wgsl_full(&ctx, include_str!("raster_shader.wgsl"));
        let raster_shader_2d = compile_wgsl_full(&ctx, include_str!("raster_shader_2d.wgsl"));
        let nv12_shader = compile_wgsl_full(&ctx, include_str!("rgba_to_nv12.wgsl"));
        let video_nv12_shader = compile_wgsl_full(&ctx, include_str!("rgba_to_nv12_image.wgsl"));
        let video_nv12_downsample_shader =
            compile_wgsl_full(&ctx, include_str!("downsample_to_nv12_image.wgsl"));
        let yuv444p_shader = compile_wgsl_full(&ctx, include_str!("rgba_to_yuv444p.wgsl"));
        let composite_shader = compile_wgsl_full(&ctx, include_str!("composite_shader.wgsl"));
        let downsample_shader = compile_wgsl_full(&ctx, include_str!("downsample_shader.wgsl"));

        let composite_bindings = [
            vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: 1,
                descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            },
        ];
        let composite_layout_info = vk::DescriptorSetLayoutCreateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            binding_count: composite_bindings.len() as u32,
            p_bindings: composite_bindings.as_ptr(),
            ..Default::default()
        };
        let composite_descriptor_set_layout = unsafe {
            ctx.device
                .create_descriptor_set_layout(&composite_layout_info, None)
                .unwrap()
        };

        let nv12_bindings = [
            vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: 2,
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            },
        ];
        let nv12_layout_info = vk::DescriptorSetLayoutCreateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            binding_count: nv12_bindings.len() as u32,
            p_bindings: nv12_bindings.as_ptr(),
            ..Default::default()
        };
        let nv12_descriptor_set_layout = unsafe {
            ctx.device
                .create_descriptor_set_layout(&nv12_layout_info, None)
                .unwrap()
        };

        let video_nv12_bindings = [
            vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: 1,
                descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: 2,
                descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            },
        ];
        let video_nv12_layout_info = vk::DescriptorSetLayoutCreateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            binding_count: video_nv12_bindings.len() as u32,
            p_bindings: video_nv12_bindings.as_ptr(),
            ..Default::default()
        };
        let video_nv12_descriptor_set_layout = unsafe {
            ctx.device
                .create_descriptor_set_layout(&video_nv12_layout_info, None)
                .unwrap()
        };

        let nv12_pipeline_layout_info = vk::PipelineLayoutCreateInfo {
            s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
            set_layout_count: 1,
            p_set_layouts: &nv12_descriptor_set_layout,
            ..Default::default()
        };
        let nv12_pipeline_layout = unsafe {
            ctx.device
                .create_pipeline_layout(&nv12_pipeline_layout_info, None)
                .unwrap()
        };

        let main_name = std::ffi::CString::new("main").unwrap();
        let nv12_stage = vk::PipelineShaderStageCreateInfo {
            s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage: vk::ShaderStageFlags::COMPUTE,
            module: nv12_shader,
            p_name: main_name.as_ptr(),
            ..Default::default()
        };
        let nv12_pipeline_info = vk::ComputePipelineCreateInfo {
            s_type: vk::StructureType::COMPUTE_PIPELINE_CREATE_INFO,
            stage: nv12_stage,
            layout: nv12_pipeline_layout,
            ..Default::default()
        };
        let nv12_pipeline = unsafe {
            ctx.device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&nv12_pipeline_info),
                    None,
                )
                .unwrap()[0]
        };

        let video_nv12_pipeline_layout_info = vk::PipelineLayoutCreateInfo {
            s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
            set_layout_count: 1,
            p_set_layouts: &video_nv12_descriptor_set_layout,
            ..Default::default()
        };
        let video_nv12_pipeline_layout = unsafe {
            ctx.device
                .create_pipeline_layout(&video_nv12_pipeline_layout_info, None)
                .unwrap()
        };
        let video_nv12_stage = vk::PipelineShaderStageCreateInfo {
            s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage: vk::ShaderStageFlags::COMPUTE,
            module: video_nv12_shader,
            p_name: main_name.as_ptr(),
            ..Default::default()
        };
        let video_nv12_pipeline_info = vk::ComputePipelineCreateInfo {
            s_type: vk::StructureType::COMPUTE_PIPELINE_CREATE_INFO,
            stage: video_nv12_stage,
            layout: video_nv12_pipeline_layout,
            ..Default::default()
        };
        let video_nv12_pipeline = unsafe {
            ctx.device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&video_nv12_pipeline_info),
                    None,
                )
                .unwrap()[0]
        };
        let video_nv12_downsample_stage = vk::PipelineShaderStageCreateInfo {
            module: video_nv12_downsample_shader,
            ..video_nv12_stage
        };
        let video_nv12_downsample_pipeline_info = vk::ComputePipelineCreateInfo {
            stage: video_nv12_downsample_stage,
            ..video_nv12_pipeline_info
        };
        let video_nv12_downsample_pipeline = unsafe {
            ctx.device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&video_nv12_downsample_pipeline_info),
                    None,
                )
                .unwrap()[0]
        };

        let yuv444p_stage = vk::PipelineShaderStageCreateInfo {
            s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage: vk::ShaderStageFlags::COMPUTE,
            module: yuv444p_shader,
            p_name: main_name.as_ptr(),
            ..Default::default()
        };
        let yuv444p_pipeline_info = vk::ComputePipelineCreateInfo {
            s_type: vk::StructureType::COMPUTE_PIPELINE_CREATE_INFO,
            stage: yuv444p_stage,
            layout: nv12_pipeline_layout,
            ..Default::default()
        };
        let yuv444p_pipeline = unsafe {
            ctx.device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&yuv444p_pipeline_info),
                    None,
                )
                .unwrap()[0]
        };

        let composite_pipeline_layout_info = vk::PipelineLayoutCreateInfo {
            s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
            set_layout_count: 1,
            p_set_layouts: &composite_descriptor_set_layout,
            ..Default::default()
        };
        let composite_pipeline_layout = unsafe {
            ctx.device
                .create_pipeline_layout(&composite_pipeline_layout_info, None)
                .unwrap()
        };

        let composite_stage = vk::PipelineShaderStageCreateInfo {
            s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage: vk::ShaderStageFlags::COMPUTE,
            module: composite_shader,
            p_name: main_name.as_ptr(),
            ..Default::default()
        };

        let composite_pipeline_info = vk::ComputePipelineCreateInfo {
            s_type: vk::StructureType::COMPUTE_PIPELINE_CREATE_INFO,
            stage: composite_stage,
            layout: composite_pipeline_layout,
            ..Default::default()
        };
        let composite_pipeline = unsafe {
            ctx.device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&composite_pipeline_info),
                    None,
                )
                .unwrap()[0]
        };
        let downsample_stage = vk::PipelineShaderStageCreateInfo {
            s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage: vk::ShaderStageFlags::COMPUTE,
            module: downsample_shader,
            p_name: main_name.as_ptr(),
            ..Default::default()
        };
        let downsample_pipeline_info = vk::ComputePipelineCreateInfo {
            s_type: vk::StructureType::COMPUTE_PIPELINE_CREATE_INFO,
            stage: downsample_stage,
            layout: composite_pipeline_layout,
            ..Default::default()
        };
        let downsample_pipeline = unsafe {
            ctx.device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&downsample_pipeline_info),
                    None,
                )
                .unwrap()[0]
        };

        let compute_bindings = [
            vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: 1,
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE
                    | vk::ShaderStageFlags::VERTEX
                    | vk::ShaderStageFlags::FRAGMENT,
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: 2,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER_DYNAMIC,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            },
        ];

        let compute_layout_info = vk::DescriptorSetLayoutCreateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            binding_count: compute_bindings.len() as u32,
            p_bindings: compute_bindings.as_ptr(),
            ..Default::default()
        };
        let compute_descriptor_set_layout = unsafe {
            ctx.device
                .create_descriptor_set_layout(&compute_layout_info, None)
                .unwrap()
        };

        let raster_bindings = [vk::DescriptorSetLayoutBinding {
            binding: 1,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            ..Default::default()
        }];
        let raster_layout_info = vk::DescriptorSetLayoutCreateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            binding_count: raster_bindings.len() as u32,
            p_bindings: raster_bindings.as_ptr(),
            ..Default::default()
        };
        let raster_descriptor_set_layout = unsafe {
            ctx.device
                .create_descriptor_set_layout(&raster_layout_info, None)
                .unwrap()
        };

        let compute_pipeline_layout_info = vk::PipelineLayoutCreateInfo {
            s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
            set_layout_count: 1,
            p_set_layouts: &compute_descriptor_set_layout,
            ..Default::default()
        };
        let compute_pipeline_layout = unsafe {
            ctx.device
                .create_pipeline_layout(&compute_pipeline_layout_info, None)
                .unwrap()
        };

        let main_name = std::ffi::CString::new("main").unwrap();
        let compute_stage = vk::PipelineShaderStageCreateInfo {
            s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage: vk::ShaderStageFlags::COMPUTE,
            module: compute_shader,
            p_name: main_name.as_ptr(),
            ..Default::default()
        };

        let compute_pipeline_info = vk::ComputePipelineCreateInfo {
            s_type: vk::StructureType::COMPUTE_PIPELINE_CREATE_INFO,
            stage: compute_stage,
            layout: compute_pipeline_layout,
            ..Default::default()
        };
        let compute_pipeline = unsafe {
            ctx.device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&compute_pipeline_info),
                    None,
                )
                .unwrap()[0]
        };

        let raster_pipeline_layout_info = vk::PipelineLayoutCreateInfo {
            s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
            set_layout_count: 1,
            p_set_layouts: &raster_descriptor_set_layout,
            ..Default::default()
        };
        let raster_pipeline_layout = unsafe {
            ctx.device
                .create_pipeline_layout(&raster_pipeline_layout_info, None)
                .unwrap()
        };

        let raster_descriptor_set_layout_bindings_2d = [vk::DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            p_immutable_samplers: std::ptr::null(),
            ..Default::default()
        }];
        let raster_descriptor_set_layout_info_2d = vk::DescriptorSetLayoutCreateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            binding_count: raster_descriptor_set_layout_bindings_2d.len() as u32,
            p_bindings: raster_descriptor_set_layout_bindings_2d.as_ptr(),
            ..Default::default()
        };
        let raster_descriptor_set_layout_2d = unsafe {
            ctx.device
                .create_descriptor_set_layout(&raster_descriptor_set_layout_info_2d, None)
                .unwrap()
        };

        let raster_pipeline_layout_info_2d = vk::PipelineLayoutCreateInfo {
            s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
            set_layout_count: 1,
            p_set_layouts: &raster_descriptor_set_layout_2d,
            ..Default::default()
        };
        let raster_pipeline_layout_2d = unsafe {
            ctx.device
                .create_pipeline_layout(&raster_pipeline_layout_info_2d, None)
                .unwrap()
        };

        let vertex_input_binding = vk::VertexInputBindingDescription {
            binding: 0,
            stride: std::mem::size_of::<Vertex>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        };

        let vertex_input_attributes = [
            vk::VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 0,
            },
            vk::VertexInputAttributeDescription {
                location: 1,
                binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 12,
            },
            vk::VertexInputAttributeDescription {
                location: 2,
                binding: 0,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: 24,
            },
        ];

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            vertex_binding_description_count: 1,
            p_vertex_binding_descriptions: &vertex_input_binding,
            vertex_attribute_description_count: vertex_input_attributes.len() as u32,
            p_vertex_attribute_descriptions: vertex_input_attributes.as_ptr(),
            ..Default::default()
        };

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            primitive_restart_enable: vk::FALSE,
            ..Default::default()
        };

        let viewport_state = vk::PipelineViewportStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            viewport_count: 1,
            scissor_count: 1,
            ..Default::default()
        };

        let rasterizer = vk::PipelineRasterizationStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            depth_clamp_enable: vk::FALSE,
            rasterizer_discard_enable: vk::FALSE,
            polygon_mode: vk::PolygonMode::FILL,
            line_width: 1.0,
            cull_mode: vk::CullModeFlags::NONE,
            front_face: vk::FrontFace::CLOCKWISE, // naga flips Y, which reverses winding order
            depth_bias_enable: vk::FALSE,
            ..Default::default()
        };

        let multisampling = vk::PipelineMultisampleStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            sample_shading_enable: vk::FALSE,
            rasterization_samples: sample_count,
            ..Default::default()
        };

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            depth_test_enable: vk::TRUE,
            depth_write_enable: vk::TRUE,
            depth_compare_op: vk::CompareOp::LESS,
            depth_bounds_test_enable: vk::FALSE,
            stencil_test_enable: vk::FALSE,
            ..Default::default()
        };

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState {
            color_write_mask: vk::ColorComponentFlags::RGBA,
            blend_enable: vk::TRUE,
            src_color_blend_factor: vk::BlendFactor::SRC_ALPHA,
            dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
            color_blend_op: vk::BlendOp::ADD,
            src_alpha_blend_factor: vk::BlendFactor::ONE,
            dst_alpha_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
            alpha_blend_op: vk::BlendOp::ADD,
        };

        let color_blending = vk::PipelineColorBlendStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            logic_op_enable: vk::FALSE,
            attachment_count: 1,
            p_attachments: &color_blend_attachment,
            ..Default::default()
        };

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state = vk::PipelineDynamicStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            dynamic_state_count: dynamic_states.len() as u32,
            p_dynamic_states: dynamic_states.as_ptr(),
            ..Default::default()
        };

        let vs_name = std::ffi::CString::new("vs_main").unwrap();
        let fs_name = std::ffi::CString::new("fs_main").unwrap();

        let shader_stages = [
            vk::PipelineShaderStageCreateInfo {
                s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage: vk::ShaderStageFlags::VERTEX,
                module: raster_shader,
                p_name: vs_name.as_ptr(),
                ..Default::default()
            },
            vk::PipelineShaderStageCreateInfo {
                s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage: vk::ShaderStageFlags::FRAGMENT,
                module: raster_shader,
                p_name: fs_name.as_ptr(),
                ..Default::default()
            },
        ];
        let raster_color_formats = [vk::Format::R8G8B8A8_UNORM];
        let pipeline_rendering_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&raster_color_formats)
            .depth_attachment_format(vk::Format::D32_SFLOAT);

        let raster_pipeline_info = vk::GraphicsPipelineCreateInfo {
            s_type: vk::StructureType::GRAPHICS_PIPELINE_CREATE_INFO,
            p_next: (&pipeline_rendering_info as *const vk::PipelineRenderingCreateInfo).cast(),
            stage_count: shader_stages.len() as u32,
            p_stages: shader_stages.as_ptr(),
            p_vertex_input_state: &vertex_input_info,
            p_input_assembly_state: &input_assembly,
            p_viewport_state: &viewport_state,
            p_rasterization_state: &rasterizer,
            p_multisample_state: &multisampling,
            p_depth_stencil_state: &depth_stencil,
            p_color_blend_state: &color_blending,
            p_dynamic_state: &dynamic_state,
            layout: raster_pipeline_layout,
            render_pass: vk::RenderPass::null(),
            ..Default::default()
        };

        let raster_pipeline = unsafe {
            ctx.device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&raster_pipeline_info),
                    None,
                )
                .unwrap()[0]
        };

        let vertex_binding_descriptions_2d = [
            vk::VertexInputBindingDescription {
                binding: 0,
                stride: std::mem::size_of::<Vertex2D>() as u32,
                input_rate: vk::VertexInputRate::VERTEX,
            },
            vk::VertexInputBindingDescription {
                binding: 1,
                stride: std::mem::size_of::<Instance2D>() as u32,
                input_rate: vk::VertexInputRate::INSTANCE,
            },
        ];
        let vertex_attribute_descriptions_2d = [
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: memoffset::offset_of!(Vertex2D, position) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding: 1,
                location: 1,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: memoffset::offset_of!(Instance2D, model_0) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding: 1,
                location: 2,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: memoffset::offset_of!(Instance2D, model_1) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding: 1,
                location: 3,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: memoffset::offset_of!(Instance2D, model_2) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding: 1,
                location: 4,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: memoffset::offset_of!(Instance2D, model_3) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding: 1,
                location: 5,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: memoffset::offset_of!(Instance2D, color) as u32,
            },
        ];
        let vertex_input_info_2d = vk::PipelineVertexInputStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            vertex_binding_description_count: vertex_binding_descriptions_2d.len() as u32,
            p_vertex_binding_descriptions: vertex_binding_descriptions_2d.as_ptr(),
            vertex_attribute_description_count: vertex_attribute_descriptions_2d.len() as u32,
            p_vertex_attribute_descriptions: vertex_attribute_descriptions_2d.as_ptr(),
            ..Default::default()
        };

        let shader_stages_2d = [
            vk::PipelineShaderStageCreateInfo {
                s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage: vk::ShaderStageFlags::VERTEX,
                module: raster_shader_2d,
                p_name: vs_name.as_ptr(),
                ..Default::default()
            },
            vk::PipelineShaderStageCreateInfo {
                s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage: vk::ShaderStageFlags::FRAGMENT,
                module: raster_shader_2d,
                p_name: fs_name.as_ptr(),
                ..Default::default()
            },
        ];

        let depth_stencil_2d = vk::PipelineDepthStencilStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            depth_test_enable: vk::FALSE, // 2D overlays without depth testing (painter's algorithm)
            depth_write_enable: vk::FALSE,
            depth_compare_op: vk::CompareOp::ALWAYS,
            depth_bounds_test_enable: vk::FALSE,
            stencil_test_enable: vk::FALSE,
            ..Default::default()
        };

        let raster_pipeline_info_2d = vk::GraphicsPipelineCreateInfo {
            s_type: vk::StructureType::GRAPHICS_PIPELINE_CREATE_INFO,
            p_next: (&pipeline_rendering_info as *const vk::PipelineRenderingCreateInfo).cast(),
            stage_count: shader_stages_2d.len() as u32,
            p_stages: shader_stages_2d.as_ptr(),
            p_vertex_input_state: &vertex_input_info_2d,
            p_input_assembly_state: &input_assembly,
            p_viewport_state: &viewport_state,
            p_rasterization_state: &rasterizer,
            p_multisample_state: &multisampling,
            p_depth_stencil_state: &depth_stencil_2d,
            p_color_blend_state: &color_blending,
            p_dynamic_state: &dynamic_state,
            layout: raster_pipeline_layout_2d,
            render_pass: vk::RenderPass::null(),
            ..Default::default()
        };

        let raster_pipeline_2d = unsafe {
            ctx.device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&raster_pipeline_info_2d),
                    None,
                )
                .unwrap()[0]
        };
        let depthless_pipeline_rendering_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&raster_color_formats);
        let raster_pipeline_info_2d_depthless = vk::GraphicsPipelineCreateInfo {
            p_next: (&depthless_pipeline_rendering_info as *const vk::PipelineRenderingCreateInfo)
                .cast(),
            ..raster_pipeline_info_2d
        };
        let raster_pipeline_2d_depthless = unsafe {
            ctx.device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&raster_pipeline_info_2d_depthless),
                    None,
                )
                .unwrap()[0]
        };

        let descriptor_pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 80,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLED_IMAGE,
                descriptor_count: 20,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 40,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 40,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
                descriptor_count: 20,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER_DYNAMIC,
                descriptor_count: 20,
            },
        ];
        let descriptor_pool_info = vk::DescriptorPoolCreateInfo {
            s_type: vk::StructureType::DESCRIPTOR_POOL_CREATE_INFO,
            pool_size_count: descriptor_pool_sizes.len() as u32,
            p_pool_sizes: descriptor_pool_sizes.as_ptr(),
            max_sets: 32,
            flags: vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET,
            ..Default::default()
        };
        let descriptor_pool = unsafe {
            ctx.device
                .create_descriptor_pool(&descriptor_pool_info, None)
                .unwrap()
        };

        let limits = unsafe {
            ctx.instance
                .get_physical_device_properties(ctx.physical_device)
                .limits
        };
        let uniform_alignment = limits.min_uniform_buffer_offset_alignment.max(1);
        let storage_alignment = limits.min_storage_buffer_offset_alignment.max(1);
        let vertex_buffer_stride = (std::mem::size_of::<Vertex>() * 1_000_000) as u64;
        let index_buffer_stride = (std::mem::size_of::<u32>() * 3_000_000) as u64;
        let camera_buffer_stride = align_up(
            std::mem::size_of::<CameraUniform>() as u64,
            uniform_alignment,
        );
        let primitive_buffer_stride = align_up(
            (std::mem::size_of::<PrimitiveData3D>() * 10_000) as u64,
            storage_alignment,
        );
        let static_vertex_buffer_2d_size = (std::mem::size_of::<Vertex2D>() * 1_000_000) as u64;
        let static_index_buffer_2d_size = (std::mem::size_of::<u32>() * 3_000_000) as u64;
        let vertex_staging_buffer_2d_stride = static_vertex_buffer_2d_size;
        let index_staging_buffer_2d_stride = static_index_buffer_2d_size;
        let instance_buffer_2d_stride = (std::mem::size_of::<Instance2D>() * 100_000) as u64;
        let camera_buffer_2d_stride = align_up(
            std::mem::size_of::<CameraUniform2D>() as u64,
            uniform_alignment,
        );
        let frame_count = RENDER_FRAME_COUNT as u64;

        let vertex_buffer = Buffer::new(
            &ctx,
            vertex_buffer_stride * frame_count,
            vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            gpu_allocator::MemoryLocation::CpuToGpu,
        );
        let index_buffer = Buffer::new(
            &ctx,
            index_buffer_stride * frame_count,
            vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            gpu_allocator::MemoryLocation::CpuToGpu,
        );
        let camera_buffer = Buffer::new(
            &ctx,
            camera_buffer_stride * frame_count,
            vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            gpu_allocator::MemoryLocation::CpuToGpu,
        );
        let buffer_3d = Buffer::new(
            &ctx,
            primitive_buffer_stride * frame_count,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            gpu_allocator::MemoryLocation::CpuToGpu,
        );
        let nv12_constants_buffer = Buffer::new(
            &ctx,
            std::mem::size_of::<Nv12Constants>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            gpu_allocator::MemoryLocation::CpuToGpu,
        );

        let vertex_buffer_2d = Buffer::new(
            &ctx,
            static_vertex_buffer_2d_size,
            vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            gpu_allocator::MemoryLocation::GpuOnly,
        );
        let index_buffer_2d = Buffer::new(
            &ctx,
            static_index_buffer_2d_size,
            vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            gpu_allocator::MemoryLocation::GpuOnly,
        );
        let vertex_staging_buffer_2d = Buffer::new(
            &ctx,
            vertex_staging_buffer_2d_stride * frame_count,
            vk::BufferUsageFlags::TRANSFER_SRC,
            gpu_allocator::MemoryLocation::CpuToGpu,
        );
        let index_staging_buffer_2d = Buffer::new(
            &ctx,
            index_staging_buffer_2d_stride * frame_count,
            vk::BufferUsageFlags::TRANSFER_SRC,
            gpu_allocator::MemoryLocation::CpuToGpu,
        );
        let instance_buffer_2d = Buffer::new(
            &ctx,
            instance_buffer_2d_stride * frame_count,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            gpu_allocator::MemoryLocation::CpuToGpu,
        );
        let camera_buffer_2d = Buffer::new(
            &ctx,
            camera_buffer_2d_stride * frame_count,
            vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            gpu_allocator::MemoryLocation::CpuToGpu,
        );

        let command_pool_info = vk::CommandPoolCreateInfo {
            s_type: vk::StructureType::COMMAND_POOL_CREATE_INFO,
            queue_family_index: ctx.queue_family_index,
            flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            ..Default::default()
        };

        let frame_data = std::array::from_fn(|_| {
            let command_pool = unsafe {
                ctx.device
                    .create_command_pool(&command_pool_info, None)
                    .unwrap()
            };
            let alloc_info = vk::CommandBufferAllocateInfo {
                s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
                command_pool,
                level: vk::CommandBufferLevel::PRIMARY,
                command_buffer_count: 1,
                ..Default::default()
            };
            let command_buffer =
                unsafe { ctx.device.allocate_command_buffers(&alloc_info).unwrap()[0] };
            let fence_info = vk::FenceCreateInfo {
                s_type: vk::StructureType::FENCE_CREATE_INFO,
                flags: vk::FenceCreateFlags::SIGNALED,
                ..Default::default()
            };
            let fence = unsafe { ctx.device.create_fence(&fence_info, None).unwrap() };
            let query_pool_info = vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::TIMESTAMP)
                .query_count(GPU_TIMESTAMP_COUNT);
            let query_pool = unsafe {
                ctx.device
                    .create_query_pool(&query_pool_info, None)
                    .unwrap()
            };
            FrameData {
                command_pool,
                command_buffer,
                fence,
                query_pool,
                timestamps_pending: false,
                profiled_plan: FrameExecutionPlan::Empty,
                profiled_geometry_upload: false,
                profiled_postprocess: false,
                profiled_output: false,
            }
        });

        unsafe {
            ctx.device.destroy_shader_module(compute_shader, None);
            ctx.device.destroy_shader_module(raster_shader, None);
            ctx.device.destroy_shader_module(raster_shader_2d, None);
            ctx.device.destroy_shader_module(nv12_shader, None);
            ctx.device.destroy_shader_module(video_nv12_shader, None);
            ctx.device
                .destroy_shader_module(video_nv12_downsample_shader, None);
            ctx.device.destroy_shader_module(yuv444p_shader, None);
            ctx.device.destroy_shader_module(composite_shader, None);
            ctx.device.destroy_shader_module(downsample_shader, None);
        }

        Self {
            ctx,
            descriptor_pool,
            compute_descriptor_set_layout,
            raster_descriptor_set_layout,
            raster_descriptor_set_layout_2d,
            composite_descriptor_set_layout,
            nv12_descriptor_set_layout,
            video_nv12_descriptor_set_layout,
            compute_pipeline_layout,
            compute_pipeline,
            composite_pipeline_layout,
            composite_pipeline,
            downsample_pipeline,
            nv12_pipeline_layout,
            nv12_pipeline,
            video_nv12_pipeline_layout,
            video_nv12_pipeline,
            video_nv12_downsample_pipeline,
            yuv444p_pipeline,
            raster_pipeline_layout,
            raster_pipeline,
            vertex_buffer,
            index_buffer,
            camera_buffer,
            buffer_3d,
            nv12_constants_buffer,
            raster_pipeline_layout_2d,
            raster_pipeline_2d,
            raster_pipeline_2d_depthless,
            vertex_buffer_2d,
            index_buffer_2d,
            vertex_staging_buffer_2d,
            index_staging_buffer_2d,
            instance_buffer_2d,
            camera_buffer_2d,
            vertex_buffer_stride,
            index_buffer_stride,
            camera_buffer_stride,
            primitive_buffer_stride,
            vertex_staging_buffer_2d_stride,
            index_staging_buffer_2d_stride,
            instance_buffer_2d_stride,
            camera_buffer_2d_stride,
            mesh_cache_2d: HashMap::new(),
            static_vertex_buffer_2d_used: 0,
            static_index_buffer_2d_used: 0,
            last_stats: RendererStats::default(),
            gpu_profiling: false,
            last_gpu_timings: None,
            frame_data,
            cache: std::sync::Mutex::new(None),
            msaa_samples: sample_count.as_raw(),
            ssaa_factor,
        }
    }

    pub fn last_stats(&self) -> RendererStats {
        self.last_stats
    }

    pub fn set_gpu_profiling(&mut self, enabled: bool) {
        self.gpu_profiling = enabled;
        if !enabled {
            self.last_gpu_timings = None;
        }
    }

    pub fn last_gpu_timings(&self) -> Option<GpuPassTimings> {
        self.last_gpu_timings
    }

    pub fn render_scene(
        &mut self,
        scene: &crate::Scene,
        scene_config: &crate::SceneConfig,
        output: Option<&mut [u8]>,
    ) {
        self.render_scene_with_outputs(scene, scene_config, output, RenderOutputs::CPU_READBACKS);
    }

    pub fn render_scene_with_outputs(
        &mut self,
        scene: &crate::Scene,
        scene_config: &crate::SceneConfig,
        output: Option<&mut [u8]>,
        mut outputs: RenderOutputs,
    ) {
        outputs.cpu_rgba |= output.is_some();
        let output_w = scene_config.output_width as f32;
        let output_h = scene_config.output_height as f32;

        let (has_clip, clip_x, clip_y, clip_w, clip_h) = match scene.clip_rect {
            Some(crate::ClipRect::Pixel(x, y, w, h)) => {
                (true, x as f32, y as f32, w as f32, h as f32)
            }
            Some(crate::ClipRect::Logical(cx, cy, w, h)) => {
                let (o_left, o_right, o_bottom, o_top, _, _) = scene.camera.ortho_params();
                let log_w = o_right - o_left;
                let log_h = o_top - o_bottom;

                let tl_x = cx - w / 2.0;
                let tl_y = cy + h / 2.0;

                let norm_x = (tl_x - o_left) / log_w;
                let norm_y = (o_top - tl_y) / log_h;
                let norm_w = w / log_w;
                let norm_h = h / log_h;

                (
                    true,
                    norm_x * output_w,
                    norm_y * output_h,
                    norm_w * output_w,
                    norm_h * output_h,
                )
            }
            None => (false, 0.0, 0.0, 0.0, 0.0),
        };

        let mut primitives_3d = Vec::new();
        let mut mesh_vertices = Vec::new();
        let mut mesh_indices = Vec::new();

        let mut mesh_submissions_2d = Vec::new();

        struct VulkanDataCollector<'a> {
            primitives_3d: &'a mut Vec<PrimitiveData3D>,
            mesh_vertices: &'a mut Vec<Vertex>,
            mesh_indices: &'a mut Vec<u32>,
            mesh_submissions_2d: &'a mut Vec<Mesh2DSubmission>,
        }

        impl<'a> crate::mobjects::RenderVisitor for VulkanDataCollector<'a> {
            fn push_mesh_2d(
                &mut self,
                mesh: &crate::mobjects::mesh_2d::TriangleMesh2D,
                transform: nalgebra::Matrix4<crate::GMFloat>,
            ) {
                self.mesh_submissions_2d.push(Mesh2DSubmission {
                    geometry: mesh.geometry(),
                    instance: Instance2D::new(transform, mesh.color()),
                });
            }

            fn push_mesh_3d(
                &mut self,
                mesh: &crate::mobjects::mesh_3d::TriangleMesh3D,
                transform: nalgebra::Matrix4<crate::GMFloat>,
            ) {
                let base_index = self.mesh_vertices.len() as u32;
                for v in &mesh.vertices {
                    let pos = nalgebra::Point3::new(
                        v.position[0] as crate::GMFloat,
                        v.position[1] as crate::GMFloat,
                        v.position[2] as crate::GMFloat,
                    );
                    let t_pos = transform.transform_point(&pos);
                    let n = nalgebra::Vector3::new(
                        v.normal[0] as crate::GMFloat,
                        v.normal[1] as crate::GMFloat,
                        v.normal[2] as crate::GMFloat,
                    );
                    let t_n = transform.transform_vector(&n).normalize();
                    self.mesh_vertices.push(Vertex {
                        position: [t_pos.x as f32, t_pos.y as f32, t_pos.z as f32],
                        normal: [t_n.x as f32, t_n.y as f32, t_n.z as f32],
                        color: v.color,
                    });
                }
                for i in &mesh.indices {
                    self.mesh_indices.push(*i + base_index);
                }
            }

            fn push_object_3d(
                &mut self,
                obj: &dyn crate::mobjects::object_3d::Object3D,
                transform: nalgebra::Matrix4<crate::GMFloat>,
            ) {
                self.primitives_3d.push(obj.as_primitive_data(transform));
            }
        }

        let mut collector = VulkanDataCollector {
            primitives_3d: &mut primitives_3d,
            mesh_vertices: &mut mesh_vertices,
            mesh_indices: &mut mesh_indices,
            mesh_submissions_2d: &mut mesh_submissions_2d,
        };

        for m in &scene.mobjects {
            m.borrow()
                .submit_to_renderer(&mut collector, nalgebra::Matrix4::identity());
        }
        let mesh_batches_2d = build_ordered_mesh_2d_batches(mesh_submissions_2d);

        let camera_uniform_2d = CameraUniform2D {
            width: output_w,
            height: output_h,
            scale_factor: scene_config.scale_factor as f32,
            _pad: 0.0,
        };

        let look = scene.camera.look_at_dir();
        let camera_uniform = CameraUniform {
            pos: [
                scene.camera.position.x as f32,
                scene.camera.position.y as f32,
                scene.camera.position.z as f32,
            ],
            _padding0: 0,
            look_at: [look.x as f32, look.y as f32, look.z as f32],
            _padding1: 0,
            up: [
                scene.camera.up_dir().x as f32,
                scene.camera.up_dir().y as f32,
                scene.camera.up_dir().z as f32,
            ],
            fov: scene.camera.fov() as f32,
            width: output_w,
            height: output_h,
            proj_type: scene.camera.proj_type(),
            ortho_left: scene.camera.ortho_params().0 as f32,
            ortho_right: scene.camera.ortho_params().1 as f32,
            ortho_bottom: scene.camera.ortho_params().2 as f32,
            ortho_top: scene.camera.ortho_params().3 as f32,
            has_clip: if has_clip { 1 } else { 0 },
            clip_x,
            clip_y,
            clip_w,
            clip_h,
            aa_level: scene.aa_level,
            num_primitives: primitives_3d.len() as u32,
            _pad2: 0,
            _pad3: 0,
            proj_mat: {
                if scene.camera.proj_type() == 0 {
                    crate::camera::Projection::perspective_wgpu(
                        scene.camera.fov() as f32,
                        output_w / output_h,
                        scene.camera.perspective_params().0 as f32,
                        scene.camera.perspective_params().1 as f32,
                    )
                } else {
                    let aspect = output_w / output_h;
                    let ortho_params = scene.camera.ortho_params();
                    // ortho_params returns (left, right, bottom, top, near, far) where left/right are often without aspect ratio applied
                    // Actually, let's just use the exact params from the camera
                    crate::camera::Projection::orthographic_wgpu(
                        ortho_params.0 as f32,
                        ortho_params.1 as f32,
                        ortho_params.2 as f32,
                        ortho_params.3 as f32,
                        ortho_params.4 as f32,
                        ortho_params.5 as f32,
                    )
                }
            },
        };

        self.render(
            scene_config.output_width,
            scene_config.output_height,
            &camera_uniform,
            &camera_uniform_2d,
            &primitives_3d,
            &mesh_vertices,
            &mesh_indices,
            &mesh_batches_2d,
            output,
            outputs,
        );
    }

    fn render(
        &mut self,
        width: u32,
        height: u32,
        camera_uniform: &CameraUniform,
        camera_uniform_2d: &CameraUniform2D,
        objects_3d: &[PrimitiveData3D],
        mesh_vertices: &[Vertex],
        mesh_indices: &[u32],
        mesh_batches_2d: &[Mesh2DBatch],
        output: Option<&mut [u8]>,
        outputs: RenderOutputs,
    ) {
        let align = 256;
        let unpadded_bytes_per_row = width * 4;
        let padded_bytes_per_row = (unpadded_bytes_per_row + align - 1) & !(align - 1);

        let mut cache_guard = self.cache.lock().unwrap();
        let cache_needs_update = cache_guard
            .as_ref()
            .map_or(true, |c| c.width != width || c.height != height);

        if cache_needs_update {
            if let Some(mut old_cache) = cache_guard.take() {
                unsafe {
                    self.ctx.device.device_wait_idle().unwrap();
                }
                old_cache.destroy(&self.ctx);
            }
            self.nv12_constants_buffer.write_bytes(
                0,
                bytemuck::bytes_of(&Nv12Constants {
                    width,
                    height,
                    _padding: [0; 2],
                }),
            );

            let mut render_targets = std::array::from_fn(|_| {
                let texture = Image::new(
                    &self.ctx,
                    width,
                    height,
                    vk::Format::R8G8B8A8_UNORM,
                    vk::ImageUsageFlags::STORAGE
                        | vk::ImageUsageFlags::SAMPLED
                        | vk::ImageUsageFlags::TRANSFER_SRC
                        | vk::ImageUsageFlags::TRANSFER_DST
                        | vk::ImageUsageFlags::COLOR_ATTACHMENT,
                    vk::ImageAspectFlags::COLOR,
                    vk::SampleCountFlags::TYPE_1,
                );
                let resolved_texture = Image::new(
                    &self.ctx,
                    width * self.ssaa_factor,
                    height * self.ssaa_factor,
                    vk::Format::R8G8B8A8_UNORM,
                    vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
                    vk::ImageAspectFlags::COLOR,
                    vk::SampleCountFlags::TYPE_1,
                );
                RenderTargetSet {
                    texture,
                    texture_state: TrackedImageState::UNDEFINED,
                    resolved_texture,
                    resolved_texture_state: TrackedImageState::UNDEFINED,
                    compute_descriptor_set: vk::DescriptorSet::null(),
                    composite_descriptor_set: vk::DescriptorSet::null(),
                }
            });
            let raster_sample_count = msaa_to_vk_sample_count(self.msaa_samples);
            let msaa_texture = (raster_sample_count != vk::SampleCountFlags::TYPE_1).then(|| {
                Image::new(
                    &self.ctx,
                    width * self.ssaa_factor,
                    height * self.ssaa_factor,
                    vk::Format::R8G8B8A8_UNORM,
                    vk::ImageUsageFlags::COLOR_ATTACHMENT,
                    vk::ImageAspectFlags::COLOR,
                    raster_sample_count,
                )
            });
            let output_buffer_size = (padded_bytes_per_row * height) as u64;
            let output_buffers = [
                Buffer::new(
                    &self.ctx,
                    output_buffer_size,
                    vk::BufferUsageFlags::TRANSFER_DST,
                    gpu_allocator::MemoryLocation::GpuToCpu,
                ),
                Buffer::new(
                    &self.ctx,
                    output_buffer_size,
                    vk::BufferUsageFlags::TRANSFER_DST,
                    gpu_allocator::MemoryLocation::GpuToCpu,
                ),
                Buffer::new(
                    &self.ctx,
                    output_buffer_size,
                    vk::BufferUsageFlags::TRANSFER_DST,
                    gpu_allocator::MemoryLocation::GpuToCpu,
                ),
            ];

            let nv12_buffer_size = (width * height * 3 / 2) as u64;
            let nv12_output_buffers = [
                Buffer::new(
                    &self.ctx,
                    nv12_buffer_size,
                    vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                    gpu_allocator::MemoryLocation::GpuToCpu,
                ),
                Buffer::new(
                    &self.ctx,
                    nv12_buffer_size,
                    vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                    gpu_allocator::MemoryLocation::GpuToCpu,
                ),
                Buffer::new(
                    &self.ctx,
                    nv12_buffer_size,
                    vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                    gpu_allocator::MemoryLocation::GpuToCpu,
                ),
            ];

            let yuv444p_buffer_size = (width * height * 3) as u64;
            let yuv444p_output_buffers = [
                Buffer::new(
                    &self.ctx,
                    yuv444p_buffer_size,
                    vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                    gpu_allocator::MemoryLocation::GpuToCpu,
                ),
                Buffer::new(
                    &self.ctx,
                    yuv444p_buffer_size,
                    vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                    gpu_allocator::MemoryLocation::GpuToCpu,
                ),
                Buffer::new(
                    &self.ctx,
                    yuv444p_buffer_size,
                    vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                    gpu_allocator::MemoryLocation::GpuToCpu,
                ),
            ];

            let compute_layouts = [self.compute_descriptor_set_layout; RENDER_FRAME_COUNT];
            let alloc_info = vk::DescriptorSetAllocateInfo {
                s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
                descriptor_pool: self.descriptor_pool,
                descriptor_set_count: RENDER_FRAME_COUNT as u32,
                p_set_layouts: compute_layouts.as_ptr(),
                ..Default::default()
            };
            let compute_descriptor_sets = unsafe {
                self.ctx
                    .device
                    .allocate_descriptor_sets(&alloc_info)
                    .unwrap()
            };
            for (targets, descriptor_set) in render_targets
                .iter_mut()
                .zip(compute_descriptor_sets.into_iter())
            {
                targets.compute_descriptor_set = descriptor_set;
            }

            let alloc_info_raster = vk::DescriptorSetAllocateInfo {
                s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
                descriptor_pool: self.descriptor_pool,
                descriptor_set_count: 1,
                p_set_layouts: &self.raster_descriptor_set_layout,
                ..Default::default()
            };
            let raster_descriptor_set = unsafe {
                self.ctx
                    .device
                    .allocate_descriptor_sets(&alloc_info_raster)
                    .unwrap()[0]
            };

            let alloc_info_raster_2d = vk::DescriptorSetAllocateInfo {
                s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
                descriptor_pool: self.descriptor_pool,
                descriptor_set_count: 1,
                p_set_layouts: &self.raster_descriptor_set_layout_2d,
                ..Default::default()
            };
            let raster_descriptor_set_2d = unsafe {
                self.ctx
                    .device
                    .allocate_descriptor_sets(&alloc_info_raster_2d)
                    .unwrap()[0]
            };

            let composite_layouts = [self.composite_descriptor_set_layout; RENDER_FRAME_COUNT];
            let alloc_info_composite = vk::DescriptorSetAllocateInfo {
                s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
                descriptor_pool: self.descriptor_pool,
                descriptor_set_count: RENDER_FRAME_COUNT as u32,
                p_set_layouts: composite_layouts.as_ptr(),
                ..Default::default()
            };
            let composite_descriptor_sets = unsafe {
                self.ctx
                    .device
                    .allocate_descriptor_sets(&alloc_info_composite)
                    .unwrap()
            };
            for (targets, descriptor_set) in render_targets
                .iter_mut()
                .zip(composite_descriptor_sets.into_iter())
            {
                targets.composite_descriptor_set = descriptor_set;
            }

            let nv12_layouts = [self.nv12_descriptor_set_layout; 3];
            let nv12_alloc_info = vk::DescriptorSetAllocateInfo {
                s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
                descriptor_pool: self.descriptor_pool,
                descriptor_set_count: 3,
                p_set_layouts: nv12_layouts.as_ptr(),
                ..Default::default()
            };
            let nv12_descriptor_sets_vec = unsafe {
                self.ctx
                    .device
                    .allocate_descriptor_sets(&nv12_alloc_info)
                    .unwrap()
            };
            let nv12_descriptor_sets: [vk::DescriptorSet; 3] =
                nv12_descriptor_sets_vec.try_into().unwrap();

            let yuv444p_alloc_info = vk::DescriptorSetAllocateInfo {
                s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
                descriptor_pool: self.descriptor_pool,
                descriptor_set_count: 3,
                p_set_layouts: nv12_layouts.as_ptr(),
                ..Default::default()
            };
            let yuv444p_descriptor_sets_vec = unsafe {
                self.ctx
                    .device
                    .allocate_descriptor_sets(&yuv444p_alloc_info)
                    .unwrap()
            };
            let yuv444p_descriptor_sets: [vk::DescriptorSet; 3] =
                yuv444p_descriptor_sets_vec.try_into().unwrap();

            let mut video_nv12_slots = (0..VIDEO_NV12_IMAGE_COUNT)
                .map(|_| VideoNv12Slot::new(&self.ctx, width, height))
                .collect::<Vec<_>>();
            let video_nv12_set_layouts =
                vec![self.video_nv12_descriptor_set_layout; VIDEO_NV12_IMAGE_COUNT];
            let video_nv12_alloc_info = vk::DescriptorSetAllocateInfo {
                s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
                descriptor_pool: self.descriptor_pool,
                descriptor_set_count: VIDEO_NV12_IMAGE_COUNT as u32,
                p_set_layouts: video_nv12_set_layouts.as_ptr(),
                ..Default::default()
            };
            let video_nv12_descriptor_sets = unsafe {
                self.ctx
                    .device
                    .allocate_descriptor_sets(&video_nv12_alloc_info)
                    .unwrap()
            };
            for (slot, descriptor_set) in
                video_nv12_slots.iter_mut().zip(video_nv12_descriptor_sets)
            {
                slot.descriptor_set = descriptor_set;
            }

            let image_infos: Vec<_> = render_targets
                .iter()
                .map(|targets| vk::DescriptorImageInfo {
                    image_view: targets.texture.view,
                    image_layout: vk::ImageLayout::GENERAL,
                    ..Default::default()
                })
                .collect();
            let resolved_image_infos: Vec<_> = render_targets
                .iter()
                .map(|targets| vk::DescriptorImageInfo {
                    image_view: targets.resolved_texture.view,
                    image_layout: vk::ImageLayout::GENERAL,
                    ..Default::default()
                })
                .collect();
            let camera_buffer_info = vk::DescriptorBufferInfo {
                buffer: self.camera_buffer.vk_buffer,
                offset: 0,
                range: self.camera_buffer_stride,
                ..Default::default()
            };
            let buffer_3d_info = vk::DescriptorBufferInfo {
                buffer: self.buffer_3d.vk_buffer,
                offset: 0,
                range: self.primitive_buffer_stride,
                ..Default::default()
            };

            let camera_buffer_2d_info = vk::DescriptorBufferInfo {
                buffer: self.camera_buffer_2d.vk_buffer,
                offset: 0,
                range: self.camera_buffer_2d_stride,
                ..Default::default()
            };

            let mut write_descriptor_sets = vec![
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: raster_descriptor_set,
                    dst_binding: 1,
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
                    descriptor_count: 1,
                    p_buffer_info: &camera_buffer_info,
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: raster_descriptor_set_2d,
                    dst_binding: 0,
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
                    descriptor_count: 1,
                    p_buffer_info: &camera_buffer_2d_info,
                    ..Default::default()
                },
            ];
            for (index, targets) in render_targets.iter().enumerate() {
                write_descriptor_sets.extend([
                    vk::WriteDescriptorSet {
                        s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                        dst_set: targets.compute_descriptor_set,
                        dst_binding: 0,
                        descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                        descriptor_count: 1,
                        p_image_info: &image_infos[index],
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                        dst_set: targets.compute_descriptor_set,
                        dst_binding: 1,
                        descriptor_type: vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
                        descriptor_count: 1,
                        p_buffer_info: &camera_buffer_info,
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                        dst_set: targets.compute_descriptor_set,
                        dst_binding: 2,
                        descriptor_type: vk::DescriptorType::STORAGE_BUFFER_DYNAMIC,
                        descriptor_count: 1,
                        p_buffer_info: &buffer_3d_info,
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                        dst_set: targets.composite_descriptor_set,
                        dst_binding: 0,
                        descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                        descriptor_count: 1,
                        p_image_info: &image_infos[index],
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                        dst_set: targets.composite_descriptor_set,
                        dst_binding: 1,
                        descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                        descriptor_count: 1,
                        p_image_info: &resolved_image_infos[index],
                        ..Default::default()
                    },
                ]);
            }

            let nv12_constants_buffer_info = vk::DescriptorBufferInfo {
                buffer: self.nv12_constants_buffer.vk_buffer,
                offset: 0,
                range: vk::WHOLE_SIZE,
                ..Default::default()
            };

            let mut nv12_buffer_infos = Vec::new();
            for i in 0..3 {
                nv12_buffer_infos.push(vk::DescriptorBufferInfo {
                    buffer: nv12_output_buffers[i].vk_buffer,
                    offset: 0,
                    range: vk::WHOLE_SIZE,
                    ..Default::default()
                });
            }

            for i in 0..3 {
                write_descriptor_sets.push(vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: nv12_descriptor_sets[i],
                    dst_binding: 0,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &image_infos[i],
                    ..Default::default()
                });
                write_descriptor_sets.push(vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: nv12_descriptor_sets[i],
                    dst_binding: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: 1,
                    p_buffer_info: &nv12_buffer_infos[i],
                    ..Default::default()
                });
                write_descriptor_sets.push(vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: nv12_descriptor_sets[i],
                    dst_binding: 2,
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: 1,
                    p_buffer_info: &nv12_constants_buffer_info,
                    ..Default::default()
                });
            }

            let mut yuv444p_buffer_infos = Vec::new();
            for i in 0..3 {
                yuv444p_buffer_infos.push(vk::DescriptorBufferInfo {
                    buffer: yuv444p_output_buffers[i].vk_buffer,
                    offset: 0,
                    range: vk::WHOLE_SIZE,
                });
            }

            for i in 0..3 {
                write_descriptor_sets.push(vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: yuv444p_descriptor_sets[i],
                    dst_binding: 0,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &image_infos[i],
                    ..Default::default()
                });
                write_descriptor_sets.push(vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: yuv444p_descriptor_sets[i],
                    dst_binding: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: 1,
                    p_buffer_info: &yuv444p_buffer_infos[i],
                    ..Default::default()
                });
                write_descriptor_sets.push(vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: yuv444p_descriptor_sets[i],
                    dst_binding: 2,
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: 1,
                    p_buffer_info: &nv12_constants_buffer_info,
                    ..Default::default()
                });
            }

            let mut video_nv12_input_infos = Vec::new();
            let mut video_nv12_y_infos = Vec::new();
            let mut video_nv12_uv_infos = Vec::new();
            for slot in &video_nv12_slots {
                video_nv12_input_infos.push(vk::DescriptorImageInfo {
                    image_view: render_targets[0].texture.view,
                    image_layout: vk::ImageLayout::GENERAL,
                    ..Default::default()
                });
                video_nv12_y_infos.push(vk::DescriptorImageInfo {
                    image_view: slot.image.y_view,
                    image_layout: vk::ImageLayout::GENERAL,
                    ..Default::default()
                });
                video_nv12_uv_infos.push(vk::DescriptorImageInfo {
                    image_view: slot.image.uv_view,
                    image_layout: vk::ImageLayout::GENERAL,
                    ..Default::default()
                });
            }
            for i in 0..VIDEO_NV12_IMAGE_COUNT {
                write_descriptor_sets.push(vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: video_nv12_slots[i].descriptor_set,
                    dst_binding: 0,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &video_nv12_input_infos[i],
                    ..Default::default()
                });
                write_descriptor_sets.push(vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: video_nv12_slots[i].descriptor_set,
                    dst_binding: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &video_nv12_y_infos[i],
                    ..Default::default()
                });
                write_descriptor_sets.push(vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: video_nv12_slots[i].descriptor_set,
                    dst_binding: 2,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &video_nv12_uv_infos[i],
                    ..Default::default()
                });
            }

            unsafe {
                self.ctx
                    .device
                    .update_descriptor_sets(&write_descriptor_sets, &[]);
            }

            *cache_guard = Some(RenderCache {
                width,
                height,
                render_targets,
                msaa_texture,
                msaa_texture_state: TrackedImageState::UNDEFINED,
                msaa_depth_texture: None,
                msaa_depth_texture_state: TrackedImageState::UNDEFINED,
                output_buffers,
                nv12_output_buffers,
                nv12_descriptor_sets,
                yuv444p_output_buffers,
                yuv444p_descriptor_sets,
                video_nv12_slots,
                current_frame: 0,
                raster_descriptor_set,
                raster_descriptor_set_2d,
                padded_bytes_per_row,
                rgba_preview_buffer: vec![0; (width * height * 4) as usize],
            });
        }

        let cache = cache_guard.as_mut().unwrap();
        let frame_idx = cache.current_frame % 3;
        let video_frame_idx = cache.current_frame % cache.video_nv12_slots.len();
        if outputs.vulkan_video && cache.current_frame > 0 {
            let previous_video_frame_idx = (cache.current_frame - 1) % cache.video_nv12_slots.len();
            assert!(
                !cache.video_nv12_slots[previous_video_frame_idx].frame_available,
                "the previous Vulkan video frame must be acquired before rendering another frame"
            );
        }

        let mut mesh_2d_arena_rebuilds = 0;
        let prepared_2d = prepare_mesh_2d_batches(
            &mut self.mesh_cache_2d,
            &mut self.static_vertex_buffer_2d_used,
            &mut self.static_index_buffer_2d_used,
            self.vertex_buffer_2d.size,
            self.index_buffer_2d.size,
            self.vertex_staging_buffer_2d_stride,
            self.index_staging_buffer_2d_stride,
            self.instance_buffer_2d_stride,
            mesh_batches_2d,
        );
        let (prepared_mesh_batches_2d, geometry_uploads_2d, instances_2d) = match prepared_2d {
            Ok(prepared) => prepared,
            Err(PrepareMesh2DError::StaticArenaExhausted) => {
                unsafe {
                    self.ctx.device.device_wait_idle().unwrap();
                }
                self.mesh_cache_2d.clear();
                self.static_vertex_buffer_2d_used = 0;
                self.static_index_buffer_2d_used = 0;
                mesh_2d_arena_rebuilds = 1;
                prepare_mesh_2d_batches(
                    &mut self.mesh_cache_2d,
                    &mut self.static_vertex_buffer_2d_used,
                    &mut self.static_index_buffer_2d_used,
                    self.vertex_buffer_2d.size,
                    self.index_buffer_2d.size,
                    self.vertex_staging_buffer_2d_stride,
                    self.index_staging_buffer_2d_stride,
                    self.instance_buffer_2d_stride,
                    mesh_batches_2d,
                )
                .expect("active 2D scene exceeds a frame or persistent geometry arena")
            }
            Err(error) => panic!("2D frame preparation failed: {error:?}"),
        };
        let frame_plan = FrameExecutionPlan::build(
            !objects_3d.is_empty(),
            !mesh_indices.is_empty() || !prepared_mesh_batches_2d.is_empty(),
            self.ssaa_factor,
        );
        let fused_video_downsample = frame_plan == FrameExecutionPlan::RasterDownsample
            && self.ssaa_factor == 2
            && outputs.vulkan_video
            && !outputs.cpu_nv12
            && !outputs.cpu_yuv444p
            && !outputs.cpu_rgba;
        let runs_postprocess = matches!(
            frame_plan,
            FrameExecutionPlan::RasterDownsample | FrameExecutionPlan::SdfRasterComposite
        ) && !fused_video_downsample;
        let raster_uses_depth = !mesh_indices.is_empty();
        if raster_uses_depth && cache.msaa_depth_texture.is_none() {
            cache.msaa_depth_texture = Some(Image::new(
                &self.ctx,
                width * self.ssaa_factor,
                height * self.ssaa_factor,
                vk::Format::D32_SFLOAT,
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                vk::ImageAspectFlags::DEPTH,
                msaa_to_vk_sample_count(self.msaa_samples),
            ));
            cache.msaa_depth_texture_state = TrackedImageState::UNDEFINED;
        }

        self.last_stats = RendererStats {
            mesh_2d_draw_calls: prepared_mesh_batches_2d.len() as u32,
            mesh_2d_instances: instances_2d.len() as u32,
            mesh_2d_geometry_uploads: geometry_uploads_2d.len() as u32,
            mesh_2d_vertex_bytes_uploaded: geometry_uploads_2d
                .iter()
                .map(|upload| std::mem::size_of_val(upload.geometry.vertices()) as u64)
                .sum(),
            mesh_2d_index_bytes_uploaded: geometry_uploads_2d
                .iter()
                .map(|upload| std::mem::size_of_val(upload.geometry.indices()) as u64)
                .sum(),
            mesh_2d_arena_rebuilds,
            sdf_dispatches: frame_plan.runs_sdf() as u32,
            raster_passes: frame_plan.runs_raster() as u32,
            depth_attachment_raster_passes: raster_uses_depth as u32,
            downsample_dispatches: (frame_plan == FrameExecutionPlan::RasterDownsample
                && !fused_video_downsample) as u32,
            fused_video_downsample_dispatches: fused_video_downsample as u32,
            composite_dispatches: (frame_plan == FrameExecutionPlan::SdfRasterComposite) as u32,
            output_conversion_dispatches: outputs.cpu_nv12 as u32
                + outputs.cpu_yuv444p as u32
                + outputs.vulkan_video as u32,
            rgba_readback_copies: outputs.cpu_rgba as u32,
        };

        let gpu_profiling = self.gpu_profiling;
        let fd = &mut self.frame_data[frame_idx];
        let vertex_buffer_offset = self.vertex_buffer_stride * frame_idx as u64;
        let index_buffer_offset = self.index_buffer_stride * frame_idx as u64;
        let camera_buffer_offset = self.camera_buffer_stride * frame_idx as u64;
        let primitive_buffer_offset = self.primitive_buffer_stride * frame_idx as u64;
        let vertex_staging_buffer_2d_offset =
            self.vertex_staging_buffer_2d_stride * frame_idx as u64;
        let index_staging_buffer_2d_offset = self.index_staging_buffer_2d_stride * frame_idx as u64;
        let instance_buffer_2d_offset = self.instance_buffer_2d_stride * frame_idx as u64;
        let camera_buffer_2d_offset = self.camera_buffer_2d_stride * frame_idx as u64;
        let compute_dynamic_offsets = [camera_buffer_offset as u32, primitive_buffer_offset as u32];
        let raster_dynamic_offsets = [camera_buffer_offset as u32];
        let raster_2d_dynamic_offsets = [camera_buffer_2d_offset as u32];
        let targets = &mut cache.render_targets[frame_idx];

        unsafe {
            self.ctx
                .device
                .wait_for_fences(std::slice::from_ref(&fd.fence), true, std::u64::MAX)
                .unwrap();
            if fd.timestamps_pending {
                let mut timestamps = [0u64; GPU_TIMESTAMP_COUNT as usize];
                self.ctx
                    .device
                    .get_query_pool_results(
                        fd.query_pool,
                        0,
                        &mut timestamps,
                        vk::QueryResultFlags::TYPE_64,
                    )
                    .unwrap();
                self.last_gpu_timings = Some(GpuPassTimings::from_timestamps(
                    timestamps,
                    self.ctx.timestamp_period_ns,
                    self.ctx.timestamp_valid_bits,
                    fd.profiled_plan,
                    fd.profiled_geometry_upload,
                    fd.profiled_postprocess,
                    fd.profiled_output,
                ));
                fd.timestamps_pending = false;
            }
            self.ctx
                .device
                .reset_fences(std::slice::from_ref(&fd.fence))
                .unwrap();
            self.ctx
                .device
                .reset_command_pool(fd.command_pool, vk::CommandPoolResetFlags::empty())
                .unwrap();

            if outputs.vulkan_video {
                let input_info = vk::DescriptorImageInfo {
                    image_view: if fused_video_downsample {
                        targets.resolved_texture.view
                    } else {
                        targets.texture.view
                    },
                    image_layout: vk::ImageLayout::GENERAL,
                    ..Default::default()
                };
                let input_write = vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: cache.video_nv12_slots[video_frame_idx].descriptor_set,
                    dst_binding: 0,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &input_info,
                    ..Default::default()
                };
                self.ctx
                    .device
                    .update_descriptor_sets(std::slice::from_ref(&input_write), &[]);
            }

            self.camera_buffer
                .write_bytes(camera_buffer_offset, bytemuck::bytes_of(camera_uniform));
            self.camera_buffer_2d.write_bytes(
                camera_buffer_2d_offset,
                bytemuck::bytes_of(camera_uniform_2d),
            );
            if !objects_3d.is_empty() {
                let bytes = bytemuck::cast_slice(objects_3d);
                let len = (self.primitive_buffer_stride as usize).min(bytes.len());
                self.buffer_3d
                    .write_bytes(primitive_buffer_offset, &bytes[..len]);
            }
            if !mesh_vertices.is_empty() {
                let bytes = bytemuck::cast_slice(mesh_vertices);
                let len = (self.vertex_buffer_stride as usize).min(bytes.len());
                self.vertex_buffer
                    .write_bytes(vertex_buffer_offset, &bytes[..len]);
            }
            if !mesh_indices.is_empty() {
                let bytes = bytemuck::cast_slice(mesh_indices);
                let len = (self.index_buffer_stride as usize).min(bytes.len());
                self.index_buffer
                    .write_bytes(index_buffer_offset, &bytes[..len]);
            }
            for upload in &geometry_uploads_2d {
                self.vertex_staging_buffer_2d.write_bytes(
                    vertex_staging_buffer_2d_offset + upload.staging_vertex_offset,
                    bytemuck::cast_slice(upload.geometry.vertices()),
                );
                self.index_staging_buffer_2d.write_bytes(
                    index_staging_buffer_2d_offset + upload.staging_index_offset,
                    bytemuck::cast_slice(upload.geometry.indices()),
                );
            }
            if !instances_2d.is_empty() {
                self.instance_buffer_2d.write_bytes(
                    instance_buffer_2d_offset,
                    bytemuck::cast_slice(&instances_2d),
                );
            }

            let begin_info = vk::CommandBufferBeginInfo {
                s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
                flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                ..Default::default()
            };
            self.ctx
                .device
                .begin_command_buffer(fd.command_buffer, &begin_info)
                .unwrap();
            if gpu_profiling {
                self.ctx.device.cmd_reset_query_pool(
                    fd.command_buffer,
                    fd.query_pool,
                    0,
                    GPU_TIMESTAMP_COUNT,
                );
            }
            write_gpu_timestamp(
                &self.ctx.device,
                fd.command_buffer,
                fd.query_pool,
                0,
                gpu_profiling,
            );

            if !geometry_uploads_2d.is_empty() {
                let vertex_copies: Vec<_> = geometry_uploads_2d
                    .iter()
                    .map(|upload| vk::BufferCopy {
                        src_offset: vertex_staging_buffer_2d_offset + upload.staging_vertex_offset,
                        dst_offset: upload.device_vertex_offset,
                        size: std::mem::size_of_val(upload.geometry.vertices()) as u64,
                    })
                    .collect();
                let index_copies: Vec<_> = geometry_uploads_2d
                    .iter()
                    .map(|upload| vk::BufferCopy {
                        src_offset: index_staging_buffer_2d_offset + upload.staging_index_offset,
                        dst_offset: upload.device_index_offset,
                        size: std::mem::size_of_val(upload.geometry.indices()) as u64,
                    })
                    .collect();
                self.ctx.device.cmd_copy_buffer(
                    fd.command_buffer,
                    self.vertex_staging_buffer_2d.vk_buffer,
                    self.vertex_buffer_2d.vk_buffer,
                    &vertex_copies,
                );
                self.ctx.device.cmd_copy_buffer(
                    fd.command_buffer,
                    self.index_staging_buffer_2d.vk_buffer,
                    self.index_buffer_2d.vk_buffer,
                    &index_copies,
                );

                let geometry_barriers = [
                    vk::BufferMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::COPY)
                        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT)
                        .dst_access_mask(vk::AccessFlags2::VERTEX_ATTRIBUTE_READ)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .buffer(self.vertex_buffer_2d.vk_buffer)
                        .offset(0)
                        .size(vk::WHOLE_SIZE),
                    vk::BufferMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::COPY)
                        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2::INDEX_INPUT)
                        .dst_access_mask(vk::AccessFlags2::INDEX_READ)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .buffer(self.index_buffer_2d.vk_buffer)
                        .offset(0)
                        .size(vk::WHOLE_SIZE),
                ];
                let geometry_dependency =
                    vk::DependencyInfo::default().buffer_memory_barriers(&geometry_barriers);
                self.ctx
                    .device
                    .cmd_pipeline_barrier2(fd.command_buffer, &geometry_dependency);
            }
            write_gpu_timestamp(
                &self.ctx.device,
                fd.command_buffer,
                fd.query_pool,
                1,
                gpu_profiling,
            );

            let color_attachment_state = TrackedImageState {
                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                access: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            };
            let compute_write_state = TrackedImageState {
                layout: vk::ImageLayout::GENERAL,
                stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                access: vk::AccessFlags2::SHADER_WRITE,
            };
            let compute_read_state = TrackedImageState {
                layout: vk::ImageLayout::GENERAL,
                stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                access: vk::AccessFlags2::SHADER_READ,
            };

            if frame_plan == FrameExecutionPlan::Empty {
                let clear_state = TrackedImageState {
                    layout: vk::ImageLayout::GENERAL,
                    stage: vk::PipelineStageFlags2::CLEAR,
                    access: vk::AccessFlags2::TRANSFER_WRITE,
                };
                transition_image(
                    &self.ctx.device,
                    fd.command_buffer,
                    targets.texture.vk_image,
                    vk::ImageAspectFlags::COLOR,
                    &mut targets.texture_state,
                    clear_state,
                );
                self.ctx.device.cmd_clear_color_image(
                    fd.command_buffer,
                    targets.texture.vk_image,
                    vk::ImageLayout::GENERAL,
                    &vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 0.0],
                    },
                    &[vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    }],
                );
            }

            if frame_plan.runs_sdf() {
                transition_image(
                    &self.ctx.device,
                    fd.command_buffer,
                    targets.texture.vk_image,
                    vk::ImageAspectFlags::COLOR,
                    &mut targets.texture_state,
                    compute_write_state,
                );
                self.ctx.device.cmd_bind_pipeline(
                    fd.command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    self.compute_pipeline,
                );
                self.ctx.device.cmd_bind_descriptor_sets(
                    fd.command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    self.compute_pipeline_layout,
                    0,
                    std::slice::from_ref(&targets.compute_descriptor_set),
                    &compute_dynamic_offsets,
                );
                self.ctx.device.cmd_dispatch(
                    fd.command_buffer,
                    (width + 15) / 16,
                    (height + 15) / 16,
                    1,
                );
            }
            write_gpu_timestamp(
                &self.ctx.device,
                fd.command_buffer,
                fd.query_pool,
                2,
                gpu_profiling,
            );

            if frame_plan.runs_raster() {
                let direct = frame_plan == FrameExecutionPlan::RasterDirect;
                let (target_image, target_view, target_state) = if direct {
                    (
                        targets.texture.vk_image,
                        targets.texture.view,
                        &mut targets.texture_state,
                    )
                } else {
                    (
                        targets.resolved_texture.vk_image,
                        targets.resolved_texture.view,
                        &mut targets.resolved_texture_state,
                    )
                };
                transition_image(
                    &self.ctx.device,
                    fd.command_buffer,
                    target_image,
                    vk::ImageAspectFlags::COLOR,
                    target_state,
                    color_attachment_state,
                );
                if raster_uses_depth {
                    let depth_texture = cache
                        .msaa_depth_texture
                        .as_ref()
                        .expect("3D raster requires a depth attachment");
                    transition_image(
                        &self.ctx.device,
                        fd.command_buffer,
                        depth_texture.vk_image,
                        vk::ImageAspectFlags::DEPTH,
                        &mut cache.msaa_depth_texture_state,
                        TrackedImageState {
                            layout: vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
                            stage: vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                                | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                            access: vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
                                | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
                        },
                    );
                }

                let mut color_attachment = vk::RenderingAttachmentInfo::default()
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.0, 0.0, 0.0, 0.0],
                        },
                    });
                if let Some(msaa_texture) = &cache.msaa_texture {
                    transition_image(
                        &self.ctx.device,
                        fd.command_buffer,
                        msaa_texture.vk_image,
                        vk::ImageAspectFlags::COLOR,
                        &mut cache.msaa_texture_state,
                        color_attachment_state,
                    );
                    color_attachment = color_attachment
                        .image_view(msaa_texture.view)
                        .store_op(vk::AttachmentStoreOp::DONT_CARE)
                        .resolve_mode(vk::ResolveModeFlags::AVERAGE)
                        .resolve_image_view(target_view)
                        .resolve_image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
                } else {
                    color_attachment = color_attachment.image_view(target_view);
                }
                let depth_attachment = cache.msaa_depth_texture.as_ref().map(|texture| {
                    vk::RenderingAttachmentInfo::default()
                        .image_view(texture.view)
                        .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                        .load_op(vk::AttachmentLoadOp::CLEAR)
                        .store_op(vk::AttachmentStoreOp::DONT_CARE)
                        .clear_value(vk::ClearValue {
                            depth_stencil: vk::ClearDepthStencilValue {
                                depth: 1.0,
                                stencil: 0,
                            },
                        })
                });
                let color_attachments = [color_attachment];
                let raster_extent = vk::Extent2D {
                    width: width * self.ssaa_factor,
                    height: height * self.ssaa_factor,
                };
                let mut rendering_info = vk::RenderingInfo::default()
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: raster_extent,
                    })
                    .layer_count(1)
                    .color_attachments(&color_attachments);
                if raster_uses_depth {
                    rendering_info = rendering_info.depth_attachment(
                        depth_attachment
                            .as_ref()
                            .expect("3D raster requires a depth attachment"),
                    );
                }
                self.ctx
                    .device
                    .cmd_begin_rendering(fd.command_buffer, &rendering_info);

                self.ctx.device.cmd_set_viewport(
                    fd.command_buffer,
                    0,
                    &[vk::Viewport {
                        x: 0.0,
                        y: 0.0,
                        width: raster_extent.width as f32,
                        height: raster_extent.height as f32,
                        min_depth: 0.0,
                        max_depth: 1.0,
                    }],
                );
                self.ctx.device.cmd_set_scissor(
                    fd.command_buffer,
                    0,
                    &[vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: raster_extent,
                    }],
                );

                if !mesh_indices.is_empty() {
                    self.ctx.device.cmd_bind_pipeline(
                        fd.command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.raster_pipeline,
                    );
                    self.ctx.device.cmd_bind_descriptor_sets(
                        fd.command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.raster_pipeline_layout,
                        0,
                        std::slice::from_ref(&cache.raster_descriptor_set),
                        &raster_dynamic_offsets,
                    );
                    self.ctx.device.cmd_bind_vertex_buffers(
                        fd.command_buffer,
                        0,
                        std::slice::from_ref(&self.vertex_buffer.vk_buffer),
                        &[vertex_buffer_offset],
                    );
                    self.ctx.device.cmd_bind_index_buffer(
                        fd.command_buffer,
                        self.index_buffer.vk_buffer,
                        index_buffer_offset,
                        vk::IndexType::UINT32,
                    );
                    let indices_to_draw =
                        (mesh_indices.len() as u32).min((self.index_buffer_stride / 4) as u32);
                    self.ctx.device.cmd_draw_indexed(
                        fd.command_buffer,
                        indices_to_draw,
                        1,
                        0,
                        0,
                        0,
                    );
                }

                if !prepared_mesh_batches_2d.is_empty() {
                    self.ctx.device.cmd_bind_pipeline(
                        fd.command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        if raster_uses_depth {
                            self.raster_pipeline_2d
                        } else {
                            self.raster_pipeline_2d_depthless
                        },
                    );
                    self.ctx.device.cmd_bind_descriptor_sets(
                        fd.command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.raster_pipeline_layout_2d,
                        0,
                        std::slice::from_ref(&cache.raster_descriptor_set_2d),
                        &raster_2d_dynamic_offsets,
                    );
                    let vertex_buffers = [
                        self.vertex_buffer_2d.vk_buffer,
                        self.instance_buffer_2d.vk_buffer,
                    ];
                    self.ctx.device.cmd_bind_vertex_buffers(
                        fd.command_buffer,
                        0,
                        &vertex_buffers,
                        &[0, instance_buffer_2d_offset],
                    );
                    self.ctx.device.cmd_bind_index_buffer(
                        fd.command_buffer,
                        self.index_buffer_2d.vk_buffer,
                        0,
                        vk::IndexType::UINT32,
                    );
                    for batch in &prepared_mesh_batches_2d {
                        self.ctx.device.cmd_draw_indexed(
                            fd.command_buffer,
                            batch.index_count,
                            batch.instance_count,
                            batch.first_index,
                            batch.vertex_offset,
                            batch.first_instance,
                        );
                    }
                }
                self.ctx.device.cmd_end_rendering(fd.command_buffer);
            }
            write_gpu_timestamp(
                &self.ctx.device,
                fd.command_buffer,
                fd.query_pool,
                3,
                gpu_profiling,
            );

            if runs_postprocess {
                transition_image(
                    &self.ctx.device,
                    fd.command_buffer,
                    targets.resolved_texture.vk_image,
                    vk::ImageAspectFlags::COLOR,
                    &mut targets.resolved_texture_state,
                    compute_read_state,
                );
                transition_image(
                    &self.ctx.device,
                    fd.command_buffer,
                    targets.texture.vk_image,
                    vk::ImageAspectFlags::COLOR,
                    &mut targets.texture_state,
                    if frame_plan == FrameExecutionPlan::SdfRasterComposite {
                        TrackedImageState {
                            layout: vk::ImageLayout::GENERAL,
                            stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                            access: vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
                        }
                    } else {
                        compute_write_state
                    },
                );
                self.ctx.device.cmd_bind_pipeline(
                    fd.command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    if frame_plan == FrameExecutionPlan::SdfRasterComposite {
                        self.composite_pipeline
                    } else {
                        self.downsample_pipeline
                    },
                );
                self.ctx.device.cmd_bind_descriptor_sets(
                    fd.command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    self.composite_pipeline_layout,
                    0,
                    std::slice::from_ref(&targets.composite_descriptor_set),
                    &[],
                );
                self.ctx.device.cmd_dispatch(
                    fd.command_buffer,
                    (width + 15) / 16,
                    (height + 15) / 16,
                    1,
                );
                targets.texture_state = compute_write_state;
            }
            write_gpu_timestamp(
                &self.ctx.device,
                fd.command_buffer,
                fd.query_pool,
                4,
                gpu_profiling,
            );

            let has_compute_output =
                outputs.cpu_nv12 || outputs.cpu_yuv444p || outputs.vulkan_video;
            if has_compute_output {
                if fused_video_downsample {
                    transition_image(
                        &self.ctx.device,
                        fd.command_buffer,
                        targets.resolved_texture.vk_image,
                        vk::ImageAspectFlags::COLOR,
                        &mut targets.resolved_texture_state,
                        compute_read_state,
                    );
                } else {
                    transition_image(
                        &self.ctx.device,
                        fd.command_buffer,
                        targets.texture.vk_image,
                        vk::ImageAspectFlags::COLOR,
                        &mut targets.texture_state,
                        compute_read_state,
                    );
                }
            }

            if outputs.cpu_nv12 {
                self.ctx.device.cmd_bind_pipeline(
                    fd.command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    self.nv12_pipeline,
                );
                self.ctx.device.cmd_bind_descriptor_sets(
                    fd.command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    self.nv12_pipeline_layout,
                    0,
                    std::slice::from_ref(&cache.nv12_descriptor_sets[frame_idx]),
                    &[],
                );
                let workgroup_x = (width / 4 + 15) / 16;
                let workgroup_y = (height / 2 + 15) / 16;
                self.ctx
                    .device
                    .cmd_dispatch(fd.command_buffer, workgroup_x, workgroup_y, 1);
            }

            if outputs.cpu_yuv444p {
                self.ctx.device.cmd_bind_pipeline(
                    fd.command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    self.yuv444p_pipeline,
                );
                self.ctx.device.cmd_bind_descriptor_sets(
                    fd.command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    self.nv12_pipeline_layout,
                    0,
                    std::slice::from_ref(&cache.yuv444p_descriptor_sets[frame_idx]),
                    &[],
                );
                let workgroup_x = (width / 4 + 15) / 16;
                let workgroup_y = (height + 15) / 16;
                self.ctx
                    .device
                    .cmd_dispatch(fd.command_buffer, workgroup_x, workgroup_y, 1);
            }

            if outputs.vulkan_video {
                let video_slot = &cache.video_nv12_slots[video_frame_idx];
                let video_nv12_barrier = vk::ImageMemoryBarrier {
                    s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
                    old_layout: video_slot.layout,
                    new_layout: vk::ImageLayout::GENERAL,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    image: video_slot.image.vk_image,
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    src_access_mask: vk::AccessFlags::empty(),
                    dst_access_mask: vk::AccessFlags::SHADER_WRITE,
                    ..Default::default()
                };
                self.ctx.device.cmd_pipeline_barrier(
                    fd.command_buffer,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    std::slice::from_ref(&video_nv12_barrier),
                );
                self.ctx.device.cmd_bind_pipeline(
                    fd.command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    if fused_video_downsample {
                        self.video_nv12_downsample_pipeline
                    } else {
                        self.video_nv12_pipeline
                    },
                );
                self.ctx.device.cmd_bind_descriptor_sets(
                    fd.command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    self.video_nv12_pipeline_layout,
                    0,
                    std::slice::from_ref(&video_slot.descriptor_set),
                    &[],
                );
                let (workgroup_x, workgroup_y) = if fused_video_downsample {
                    (((width / 2) + 15) / 16, ((height / 2) + 15) / 16)
                } else {
                    ((width + 15) / 16, (height + 15) / 16)
                };
                self.ctx
                    .device
                    .cmd_dispatch(fd.command_buffer, workgroup_x, workgroup_y, 1);
                cache.video_nv12_slots[video_frame_idx].layout = vk::ImageLayout::GENERAL;
            }

            if outputs.cpu_rgba {
                transition_image(
                    &self.ctx.device,
                    fd.command_buffer,
                    targets.texture.vk_image,
                    vk::ImageAspectFlags::COLOR,
                    &mut targets.texture_state,
                    TrackedImageState {
                        layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        stage: vk::PipelineStageFlags2::COPY,
                        access: vk::AccessFlags2::TRANSFER_READ,
                    },
                );

                let buffer_row_length = cache.padded_bytes_per_row / 4;
                let copy_region = vk::BufferImageCopy {
                    buffer_offset: 0,
                    buffer_row_length,
                    buffer_image_height: height,
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                    image_extent: vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    },
                    ..Default::default()
                };

                self.ctx.device.cmd_copy_image_to_buffer(
                    fd.command_buffer,
                    targets.texture.vk_image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    cache.output_buffers[frame_idx].vk_buffer,
                    std::slice::from_ref(&copy_region),
                );
            }
            write_gpu_timestamp(
                &self.ctx.device,
                fd.command_buffer,
                fd.query_pool,
                5,
                gpu_profiling,
            );

            self.ctx
                .device
                .end_command_buffer(fd.command_buffer)
                .unwrap();

            let mut wait_infos = Vec::with_capacity(1);
            let mut signal_infos = Vec::with_capacity(1);
            if outputs.vulkan_video {
                let video_slot = &cache.video_nv12_slots[video_frame_idx];
                let (wait_value, ready_value, _) =
                    video_timeline_values(video_slot.next_ready_value);
                if let Some(wait_value) = wait_value {
                    wait_infos.push(
                        vk::SemaphoreSubmitInfo::default()
                            .semaphore(video_slot.timeline.handle())
                            .value(wait_value)
                            .stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER),
                    );
                }
                signal_infos.push(
                    vk::SemaphoreSubmitInfo::default()
                        .semaphore(video_slot.timeline.handle())
                        .value(ready_value)
                        .stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER),
                );
            }
            let command_buffer_info =
                vk::CommandBufferSubmitInfo::default().command_buffer(fd.command_buffer);
            let submit_info = vk::SubmitInfo2::default()
                .wait_semaphore_infos(&wait_infos)
                .command_buffer_infos(std::slice::from_ref(&command_buffer_info))
                .signal_semaphore_infos(&signal_infos);
            self.ctx
                .device
                .queue_submit2(self.ctx.queue, std::slice::from_ref(&submit_info), fd.fence)
                .unwrap();
            if gpu_profiling {
                fd.timestamps_pending = true;
                fd.profiled_plan = frame_plan;
                fd.profiled_geometry_upload = !geometry_uploads_2d.is_empty();
                fd.profiled_postprocess = runs_postprocess;
                fd.profiled_output = outputs.cpu_nv12
                    || outputs.cpu_yuv444p
                    || outputs.vulkan_video
                    || outputs.cpu_rgba;
            }
            if outputs.vulkan_video {
                let video_slot = &mut cache.video_nv12_slots[video_frame_idx];
                video_slot.last_ready_value = Some(video_slot.next_ready_value);
                video_slot.next_ready_value += 2;
                video_slot.frame_available = true;
            }
        }

        if let Some(out_buf) = output {
            let read_frame_idx = cache.current_frame % 3;
            let read_fd = &self.frame_data[read_frame_idx];

            unsafe {
                self.ctx
                    .device
                    .wait_for_fences(std::slice::from_ref(&read_fd.fence), true, std::u64::MAX)
                    .unwrap();
            }

            if let Some(alloc) = &cache.output_buffers[read_frame_idx].allocation {
                if let Some(mapped) = alloc.mapped_ptr() {
                    let padded_data = unsafe {
                        std::slice::from_raw_parts(
                            mapped.as_ptr() as *const u8,
                            (cache.padded_bytes_per_row * height) as usize,
                        )
                    };
                    for row in 0..height {
                        let start = (row * cache.padded_bytes_per_row) as usize;
                        let end = start + unpadded_bytes_per_row as usize;
                        let dst_start = (row * unpadded_bytes_per_row) as usize;
                        let dst_end = dst_start + unpadded_bytes_per_row as usize;
                        out_buf[dst_start..dst_end].copy_from_slice(&padded_data[start..end]);
                    }
                }
            }
        }

        cache.current_frame += 1;
    }

    pub fn get_nv12_bytes(&self) -> Option<&[u8]> {
        let guard = self.cache.lock().unwrap();
        if let Some(cache) = guard.as_ref() {
            if cache.current_frame == 0 {
                return None;
            }
            let read_frame_idx = (cache.current_frame - 1) % 3;
            let read_fd = &self.frame_data[read_frame_idx];

            unsafe {
                self.ctx
                    .device
                    .wait_for_fences(std::slice::from_ref(&read_fd.fence), true, std::u64::MAX)
                    .unwrap();
            }

            if let Some(alloc) = &cache.nv12_output_buffers[read_frame_idx].allocation {
                if let Some(mapped) = alloc.mapped_ptr() {
                    let len = (cache.width * cache.height * 3 / 2) as usize;
                    return Some(unsafe {
                        std::slice::from_raw_parts(mapped.as_ptr() as *const u8, len)
                    });
                }
            }
        }
        None
    }

    pub fn get_yuv444p_bytes(&self) -> Option<&[u8]> {
        let guard = self.cache.lock().unwrap();
        if let Some(cache) = guard.as_ref() {
            if cache.current_frame == 0 {
                return None;
            }
            let read_frame_idx = (cache.current_frame - 1) % 3;
            let read_fd = &self.frame_data[read_frame_idx];

            unsafe {
                self.ctx
                    .device
                    .wait_for_fences(std::slice::from_ref(&read_fd.fence), true, std::u64::MAX)
                    .unwrap();
            }

            if let Some(alloc) = &cache.yuv444p_output_buffers[read_frame_idx].allocation {
                if let Some(mapped) = alloc.mapped_ptr() {
                    let len = (cache.width * cache.height * 3) as usize;
                    return Some(unsafe {
                        std::slice::from_raw_parts(mapped.as_ptr() as *const u8, len)
                    });
                }
            }
        }
        None
    }

    pub fn get_vulkan_video_frame(&self) -> Option<VulkanVideoFrame> {
        let mut guard = self.cache.lock().unwrap();
        if let Some(cache) = guard.as_mut() {
            if cache.current_frame == 0 || cache.video_nv12_slots.is_empty() {
                return None;
            }
            let video_frame_idx = (cache.current_frame - 1) % cache.video_nv12_slots.len();
            let slot = &mut cache.video_nv12_slots[video_frame_idx];
            if !slot.frame_available {
                return None;
            }
            let ready_value = slot.last_ready_value?;
            let (_, _, release_value) = video_timeline_values(ready_value);
            slot.frame_available = false;
            return Some(VulkanVideoFrame::new(
                slot.image.vk_image,
                slot.image.color_view,
                slot.layout,
                slot.image.format,
                slot.image.width,
                slot.image.height,
                self.ctx.device.handle(),
                slot.timeline.clone(),
                ready_value,
                release_value,
            ));
        }
        None
    }

    pub fn get_rgba_bytes(&self) -> Option<&[u8]> {
        let mut guard = self.cache.lock().unwrap();
        if let Some(cache) = guard.as_mut() {
            if cache.current_frame == 0 {
                return None;
            }
            let read_frame_idx = (cache.current_frame - 1) % 3;
            let read_fd = &self.frame_data[read_frame_idx];

            unsafe {
                self.ctx
                    .device
                    .wait_for_fences(std::slice::from_ref(&read_fd.fence), true, std::u64::MAX)
                    .unwrap();
            }

            if let Some(alloc) = &cache.output_buffers[read_frame_idx].allocation {
                if let Some(mapped) = alloc.mapped_ptr() {
                    let height = cache.height;
                    let unpadded_bytes_per_row = cache.width * 4;
                    let padded_data = unsafe {
                        std::slice::from_raw_parts(
                            mapped.as_ptr() as *const u8,
                            (cache.padded_bytes_per_row * height) as usize,
                        )
                    };
                    for row in 0..height {
                        let start = (row * cache.padded_bytes_per_row) as usize;
                        let end = start + unpadded_bytes_per_row as usize;
                        let dst_start = (row * unpadded_bytes_per_row) as usize;
                        let dst_end = dst_start + unpadded_bytes_per_row as usize;
                        cache.rgba_preview_buffer[dst_start..dst_end]
                            .copy_from_slice(&padded_data[start..end]);
                    }
                    let ptr = cache.rgba_preview_buffer.as_ptr();
                    let len = cache.rgba_preview_buffer.len();
                    return Some(unsafe { std::slice::from_raw_parts(ptr, len) });
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use nalgebra::Matrix4;

    use super::{
        FrameExecutionPlan, Instance2D, Mesh2DSubmission, MeshGeometry2D, PrepareMesh2DError,
        build_ordered_mesh_2d_batches, prepare_mesh_2d_batches, timestamp_delta,
        video_timeline_values,
    };
    use crate::mobjects::mesh_2d::Vertex2D;

    fn triangle(offset: f32) -> Arc<MeshGeometry2D> {
        Arc::new(MeshGeometry2D::new(
            vec![
                Vertex2D {
                    position: [offset, 0.0],
                },
                Vertex2D {
                    position: [offset + 1.0, 0.0],
                },
                Vertex2D {
                    position: [offset, 1.0],
                },
            ],
            vec![0, 1, 2],
        ))
    }

    fn instance(color: [f32; 4]) -> Instance2D {
        Instance2D::new(Matrix4::identity(), color)
    }

    #[test]
    fn video_timeline_values_alternate_ready_and_release() {
        assert_eq!(video_timeline_values(1), (None, 1, 2));
        assert_eq!(video_timeline_values(3), (Some(2), 3, 4));
        assert_eq!(video_timeline_values(9), (Some(8), 9, 10));
    }

    #[test]
    #[should_panic(expected = "must be odd")]
    fn video_timeline_rejects_even_ready_values() {
        video_timeline_values(2);
    }

    #[test]
    fn frame_execution_plan_eliminates_empty_passes() {
        assert_eq!(
            FrameExecutionPlan::build(false, false, 2),
            FrameExecutionPlan::Empty
        );
        assert_eq!(
            FrameExecutionPlan::build(true, false, 2),
            FrameExecutionPlan::SdfOnly
        );
        assert_eq!(
            FrameExecutionPlan::build(false, true, 1),
            FrameExecutionPlan::RasterDirect
        );
        assert_eq!(
            FrameExecutionPlan::build(false, true, 2),
            FrameExecutionPlan::RasterDownsample
        );
        assert_eq!(
            FrameExecutionPlan::build(true, true, 1),
            FrameExecutionPlan::SdfRasterComposite
        );
        assert_eq!(
            FrameExecutionPlan::build(true, true, 2),
            FrameExecutionPlan::SdfRasterComposite
        );
    }

    #[test]
    fn timestamp_delta_handles_hardware_counter_wraparound() {
        assert_eq!(timestamp_delta(10, 25, 64), 15);
        assert_eq!(timestamp_delta(250, 5, 8), 11);
    }

    #[test]
    fn ordered_batching_only_merges_consecutive_equal_geometry() {
        let first = triangle(0.0);
        let equal_to_first = triangle(0.0);
        let second = triangle(2.0);
        let batches = build_ordered_mesh_2d_batches(vec![
            Mesh2DSubmission {
                geometry: first.clone(),
                instance: instance([1.0, 0.0, 0.0, 1.0]),
            },
            Mesh2DSubmission {
                geometry: equal_to_first,
                instance: instance([0.0, 1.0, 0.0, 1.0]),
            },
            Mesh2DSubmission {
                geometry: second,
                instance: instance([0.0, 0.0, 1.0, 1.0]),
            },
            Mesh2DSubmission {
                geometry: first,
                instance: instance([1.0, 1.0, 1.0, 1.0]),
            },
        ]);

        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0].instances.len(), 2);
        assert_eq!(batches[1].instances.len(), 1);
        assert_eq!(batches[2].instances.len(), 1);
    }

    #[test]
    fn persistent_geometry_is_only_uploaded_on_cache_miss() {
        let geometry = triangle(0.0);
        let batches = build_ordered_mesh_2d_batches(vec![
            Mesh2DSubmission {
                geometry: geometry.clone(),
                instance: instance([1.0; 4]),
            },
            Mesh2DSubmission {
                geometry,
                instance: instance([0.5; 4]),
            },
        ]);
        let mut cache = HashMap::new();
        let mut vertex_used = 0;
        let mut index_used = 0;

        let (first_draws, first_uploads, first_instances) = prepare_mesh_2d_batches(
            &mut cache,
            &mut vertex_used,
            &mut index_used,
            4096,
            4096,
            4096,
            4096,
            4096,
            &batches,
        )
        .unwrap();
        let vertex_used_after_first = vertex_used;
        let index_used_after_first = index_used;
        let (second_draws, second_uploads, second_instances) = prepare_mesh_2d_batches(
            &mut cache,
            &mut vertex_used,
            &mut index_used,
            4096,
            4096,
            4096,
            4096,
            4096,
            &batches,
        )
        .unwrap();

        assert_eq!(first_draws.len(), 1);
        assert_eq!(first_instances.len(), 2);
        assert_eq!(first_uploads.len(), 1);
        assert_eq!(second_draws.len(), 1);
        assert_eq!(second_instances.len(), 2);
        assert!(second_uploads.is_empty());
        assert_eq!(vertex_used, vertex_used_after_first);
        assert_eq!(index_used, index_used_after_first);
    }

    #[test]
    fn persistent_arena_exhaustion_is_reported_for_generation_rebuild() {
        let batches = build_ordered_mesh_2d_batches(vec![Mesh2DSubmission {
            geometry: triangle(0.0),
            instance: instance([1.0; 4]),
        }]);
        let mut cache = HashMap::new();
        let mut vertex_used = 0;
        let mut index_used = 0;

        let result = prepare_mesh_2d_batches(
            &mut cache,
            &mut vertex_used,
            &mut index_used,
            1,
            1,
            4096,
            4096,
            4096,
            &batches,
        );

        assert!(matches!(
            result,
            Err(PrepareMesh2DError::StaticArenaExhausted)
        ));
    }
}
