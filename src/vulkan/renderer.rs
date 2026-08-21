use crate::mobjects::mesh_2d::Vertex2D;
use crate::mobjects::mesh_3d::{AlphaMode3D, SurfaceMaterial, Vertex};
use crate::mobjects::object_3d::SdfPrimitive;
use crate::video_backend::vulkan_h264::VulkanVideoFrame;
use crate::vulkan::context::{TimelineSemaphore, VulkanContext};
use ash::vk;
use ash::vk::Handle;
use ash::vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_MAIN;
use std::sync::Arc;

mod frame;
mod mesh_2d;
mod output;
mod pipelines;
mod profiling;
mod record;
mod resource;
mod scene;
mod targets;
use frame::{FrameExecutionPlan, TrackedImageState, transition_image, write_gpu_timestamp};
use mesh_2d::{
    GeometryUpload2D, Instance2D, Mesh2DBatch, Mesh2DUploadPlanner, PrepareMesh2DError,
    PreparedMesh2D, PreparedMesh2DBatch,
};
pub use output::RenderOutputs;
use pipelines::PipelineSet;
use profiling::timestamp_delta;
pub use profiling::{GpuPassTimings, RendererStats};
use record::{
    CommandRecorder, GeometryUploadBuffers2D, Mesh2DBindings, Mesh2DPass, Mesh3DBindings,
    Mesh3DPass, OutputPasses, VideoOutputPass,
};
pub use resource::{Buffer, Image};
use scene::ScenePreparer;
use targets::{SurfaceComputePipelines, TargetCache, TargetCacheResources, record_surface_compute};

// The encoder keeps 8 frames in flight; one extra image prevents the renderer
// from overwriting an image before submit-side backpressure can release a slot.
const VIDEO_NV12_IMAGE_COUNT: usize = 9;
const VK_FORMAT_G8_B8R8_2PLANE_420_UNORM_RAW: i32 = 1_000_156_003;
const VK_IMAGE_USAGE_VIDEO_ENCODE_SRC_BIT_KHR_RAW: u32 = 0x0000_4000;
const VK_IMAGE_ASPECT_PLANE_0_BIT_RAW: u32 = 0x0000_0010;
const VK_IMAGE_ASPECT_PLANE_1_BIT_RAW: u32 = 0x0000_0020;
const RENDER_FRAME_COUNT: usize = 3;
const GPU_TIMESTAMP_COUNT: u32 = 6;
const MAX_SURFACE_MATERIALS: usize = 10_000;

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

fn normalized(direction: [f32; 3]) -> [f32; 3] {
    let inverse_length = 1.0
        / (direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2])
            .sqrt();
    [
        direction[0] * inverse_length,
        direction[1] * inverse_length,
        direction[2] * inverse_length,
    ]
}

fn studio_environment(direction: [f32; 3], roughness: f32) -> [f32; 4] {
    let key = normalized([-0.55, 0.65, 0.52]);
    let rim = normalized([0.72, 0.2, -0.66]);
    let ceiling = normalized([0.0, 1.0, 0.08]);
    let dot = |axis: [f32; 3]| {
        (direction[0] * axis[0] + direction[1] * axis[1] + direction[2] * axis[2]).max(0.0)
    };
    let sharpness = 2.0 + 126.0 * (1.0 - roughness).powi(4);
    let key_light = dot(key).powf(sharpness) * (7.5 - 5.5 * roughness);
    let rim_light = dot(rim).powf(sharpness * 0.55) * (3.2 - 2.0 * roughness);
    let ceiling_light = dot(ceiling).powf(4.0 + sharpness * 0.12) * 1.4;
    let horizon = (1.0 - direction[1].abs()).powf(3.0) * 0.18;
    // Ambient terms must stay channel-uniform: any per-channel gain here leaks into
    // the diffuse irradiance and tints every surface when no point light dominates.
    let ambient = 0.04 + horizon * 0.9;
    [
        ambient + key_light * 1.0 + rim_light * 0.65 + ceiling_light * 0.7,
        ambient + key_light * 0.96 + rim_light * 0.65 + ceiling_light * 0.7,
        ambient + key_light * 0.92 + rim_light * 0.65 + ceiling_light * 0.7,
        1.0,
    ]
}

fn create_studio_environment(ctx: &VulkanContext) -> (Image, vk::Sampler) {
    const WIDTH: u32 = 256;
    const HEIGHT: u32 = 128;
    const MIP_LEVELS: u32 = 9;

    let mut pixels = Vec::<f32>::new();
    let mut regions = Vec::with_capacity(MIP_LEVELS as usize);
    let mut byte_offset = 0_u64;
    for mip_level in 0..MIP_LEVELS {
        let width = (WIDTH >> mip_level).max(1);
        let height = (HEIGHT >> mip_level).max(1);
        let roughness = mip_level as f32 / (MIP_LEVELS - 1) as f32;
        regions.push(
            vk::BufferImageCopy::default()
                .buffer_offset(byte_offset)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                }),
        );
        for y in 0..height {
            let theta = (y as f32 + 0.5) / height as f32 * std::f32::consts::PI;
            let sin_theta = theta.sin();
            for x in 0..width {
                let phi = ((x as f32 + 0.5) / width as f32 - 0.5) * 2.0 * std::f32::consts::PI;
                pixels.extend(studio_environment(
                    [sin_theta * phi.cos(), theta.cos(), sin_theta * phi.sin()],
                    roughness,
                ));
            }
        }
        byte_offset += (width * height * 4 * std::mem::size_of::<f32>() as u32) as u64;
    }

    let mut staging = Buffer::new(
        ctx,
        byte_offset,
        vk::BufferUsageFlags::TRANSFER_SRC,
        gpu_allocator::MemoryLocation::CpuToGpu,
    );
    staging.write_bytes(0, bytemuck::cast_slice(&pixels));
    let image = Image::new_with_mip_levels(
        ctx,
        WIDTH,
        HEIGHT,
        vk::Format::R32G32B32A32_SFLOAT,
        vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
        vk::ImageAspectFlags::COLOR,
        vk::SampleCountFlags::TYPE_1,
        MIP_LEVELS,
    );

    let command_pool = unsafe {
        ctx.device
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(ctx.queue_family_index)
                    .flags(vk::CommandPoolCreateFlags::TRANSIENT),
                None,
            )
            .unwrap()
    };
    let command_buffer = unsafe {
        ctx.device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
            .unwrap()[0]
    };
    unsafe {
        ctx.device
            .begin_command_buffer(
                command_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .unwrap();
        let upload_barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::COPY)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .image(image.vk_image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: MIP_LEVELS,
                base_array_layer: 0,
                layer_count: 1,
            });
        ctx.device.cmd_pipeline_barrier2(
            command_buffer,
            &vk::DependencyInfo::default()
                .image_memory_barriers(std::slice::from_ref(&upload_barrier)),
        );
        ctx.device.cmd_copy_buffer_to_image(
            command_buffer,
            staging.vk_buffer,
            image.vk_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &regions,
        );
        let sample_barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COPY)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
            .dst_access_mask(vk::AccessFlags2::SHADER_READ)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image(image.vk_image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: MIP_LEVELS,
                base_array_layer: 0,
                layer_count: 1,
            });
        ctx.device.cmd_pipeline_barrier2(
            command_buffer,
            &vk::DependencyInfo::default()
                .image_memory_barriers(std::slice::from_ref(&sample_barrier)),
        );
        ctx.device.end_command_buffer(command_buffer).unwrap();
        let command_buffers = [command_buffer];
        ctx.device
            .queue_submit(
                ctx.queue,
                &[vk::SubmitInfo::default().command_buffers(&command_buffers)],
                vk::Fence::null(),
            )
            .unwrap();
        ctx.device.queue_wait_idle(ctx.queue).unwrap();
        ctx.device.destroy_command_pool(command_pool, None);
    }
    staging.destroy(ctx);

    let sampler = unsafe {
        ctx.device
            .create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::REPEAT)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .min_lod(0.0)
                    .max_lod((MIP_LEVELS - 1) as f32),
                None,
            )
            .unwrap()
    };
    (image, sampler)
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
pub struct ToneMapConstants {
    /// Raster-scale factor of the resolved image relative to the output.
    /// Analytic-AA 2D frames raster at 1x even when `ssaa_factor` is larger,
    /// so the tone-map downsample loop must use this instead of the image
    /// dimensions.
    pub factor: u32,
    pub _padding: [u32; 3],
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
    pub raster_scale: u32,
    pub has_raster_surfaces: u32,
    pub proj_mat: [f32; 16],
    pub light_pos: [f32; 3],
    pub light_intensity: f32,
    pub light_color: [f32; 3],
    pub environment_intensity: f32,
    pub environment_color: [f32; 3],
    pub environment_rotation: f32,
    pub background_color: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuSdfPrimitive {
    material_index: u32,
    shape_type: u32,
    padding: [u32; 2],
    params: [f32; 12],
}

impl GpuSdfPrimitive {
    fn encode(primitive: SdfPrimitive, material_index: u32) -> Self {
        let (shape_type, params) = match primitive {
            SdfPrimitive::Sphere { center, radius } => (
                0,
                [
                    center.x as f32,
                    center.y as f32,
                    center.z as f32,
                    radius as f32,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            ),
            SdfPrimitive::Capsule { start, end, radius } => (
                1,
                [
                    start.x as f32,
                    start.y as f32,
                    start.z as f32,
                    end.x as f32,
                    end.y as f32,
                    end.z as f32,
                    radius as f32,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            ),
            SdfPrimitive::Arrow {
                start,
                end,
                shaft_radius,
                head_radius,
                head_length,
            } => (
                2,
                [
                    start.x as f32,
                    start.y as f32,
                    start.z as f32,
                    end.x as f32,
                    end.y as f32,
                    end.z as f32,
                    shaft_radius as f32,
                    head_radius as f32,
                    head_length as f32,
                    0.0,
                    0.0,
                    0.0,
                ],
            ),
            SdfPrimitive::OrientedBox {
                center,
                half_extents,
                x_axis,
                y_axis,
            } => (
                3,
                [
                    center.x as f32,
                    center.y as f32,
                    center.z as f32,
                    half_extents.x as f32,
                    half_extents.y as f32,
                    half_extents.z as f32,
                    x_axis.x as f32,
                    x_axis.y as f32,
                    x_axis.z as f32,
                    y_axis.x as f32,
                    y_axis.y as f32,
                    y_axis.z as f32,
                ],
            ),
            SdfPrimitive::QuadraticBezier {
                start,
                control,
                end,
                radius,
            } => (
                4,
                [
                    start.x as f32,
                    start.y as f32,
                    start.z as f32,
                    control.x as f32,
                    control.y as f32,
                    control.z as f32,
                    end.x as f32,
                    end.y as f32,
                    end.z as f32,
                    radius as f32,
                    0.0,
                    0.0,
                ],
            ),
        };
        Self {
            material_index,
            shape_type,
            padding: [0; 2],
            params,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct MaterialData3D {
    base_color: [f32; 4],
    emissive: [f32; 4],
    grid_color: [f32; 4],
    surface: [f32; 4],
    grid: [f32; 4],
    grid_backface: [f32; 4],
    transmission: [f32; 4],
    absorption: [f32; 4],
    patch_corner_0: [f32; 4],
    patch_corner_1: [f32; 4],
    patch_corner_2: [f32; 4],
    patch_color: [f32; 4],
    patch_edge_color: [f32; 4],
    patch_params: [f32; 4],
}

impl From<SurfaceMaterial> for MaterialData3D {
    fn from(material: SurfaceMaterial) -> Self {
        let grid = material.spherical_grid.unwrap_or_default();
        let patch = material.spherical_patch;
        let transmission = match material.alpha_mode {
            AlphaMode3D::Opaque => None,
            AlphaMode3D::Blend(transmission) => Some(transmission),
        };
        let patch_directions = patch.map(|patch| patch.directions).unwrap_or([[0.0; 3]; 3]);
        Self {
            base_color: material.base_color,
            emissive: [
                material.emissive[0],
                material.emissive[1],
                material.emissive[2],
                material.emissive_strength,
            ],
            grid_color: grid.color,
            surface: [
                material.roughness,
                material.metallic,
                material.reflectance,
                if transmission.is_some() { 1.0 } else { 0.0 },
            ],
            grid: [
                grid.longitude_count,
                grid.latitude_count,
                grid.line_width_pixels,
                if material.spherical_grid.is_some() {
                    1.0
                } else {
                    0.0
                },
            ],
            grid_backface: [
                grid.backface_intensity,
                if material.unlit { 1.0 } else { 0.0 },
                if material.flat_shading { 1.0 } else { 0.0 },
                0.0,
            ],
            transmission: transmission
                .map(|transmission| {
                    [
                        transmission.opacity,
                        transmission.fresnel_opacity,
                        transmission.ior,
                        0.0,
                    ]
                })
                .unwrap_or([1.0, 0.0, 1.0, 0.0]),
            absorption: transmission
                .map(|transmission| {
                    [
                        transmission.absorption[0],
                        transmission.absorption[1],
                        transmission.absorption[2],
                        transmission.backface_opacity_scale,
                    ]
                })
                .unwrap_or([0.0, 0.0, 0.0, 1.0]),
            patch_corner_0: [
                patch_directions[0][0],
                patch_directions[0][1],
                patch_directions[0][2],
                0.0,
            ],
            patch_corner_1: [
                patch_directions[1][0],
                patch_directions[1][1],
                patch_directions[1][2],
                0.0,
            ],
            patch_corner_2: [
                patch_directions[2][0],
                patch_directions[2][1],
                patch_directions[2][2],
                0.0,
            ],
            patch_color: patch.map(|patch| patch.color).unwrap_or([0.0; 4]),
            patch_edge_color: patch.map(|patch| patch.edge_color).unwrap_or([0.0; 4]),
            patch_params: [
                patch
                    .map(|patch| patch.edge_width_pixels)
                    .unwrap_or_default(),
                0.0,
                0.0,
                0.0,
            ],
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Mesh3DDraw {
    first_index: u32,
    index_count: u32,
    material_index: u32,
    transparent: bool,
    view_depth: f32,
}

impl Mesh3DDraw {
    fn is_transparent(self) -> bool {
        self.transparent
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

pub struct VulkanRenderer {
    ctx: Arc<VulkanContext>,
    msaa_samples: u32,
    ssaa_factor: u32,
    environment_map: Image,
    environment_sampler: vk::Sampler,

    descriptor_pool: vk::DescriptorPool,
    pipelines: PipelineSet,
    raster_texture_set: vk::DescriptorSet,
    pub active_textures: Vec<Image>,
    texture_sampler: vk::Sampler,

    vertex_buffer: Buffer,
    index_buffer: Buffer,
    camera_buffer: Buffer,
    material_buffer_3d: Buffer,
    buffer_3d: Buffer,
    nv12_constants_buffer: Buffer,

    vertex_buffer_2d: Buffer,
    index_buffer_2d: Buffer,
    vertex_staging_buffer_2d: Buffer,
    index_staging_buffer_2d: Buffer,
    instance_buffer_2d: Buffer,
    camera_buffer_2d: Buffer,
    tone_map_factor_buffer: Buffer,
    vertex_buffer_stride: u64,
    index_buffer_stride: u64,
    camera_buffer_stride: u64,
    material_buffer_3d_stride: u64,
    primitive_buffer_stride: u64,
    vertex_staging_buffer_2d_stride: u64,
    index_staging_buffer_2d_stride: u64,
    instance_buffer_2d_stride: u64,
    camera_buffer_2d_stride: u64,
    tone_map_factor_stride: u64,
    mesh_upload_planner_2d: Mesh2DUploadPlanner,
    scene_preparer: ScenePreparer,
    last_stats: RendererStats,
    gpu_profiling: bool,
    last_gpu_timings: Option<GpuPassTimings>,
    bloom_enabled: bool,
    analytic_aa_2d: bool,

    frame_data: [FrameData; RENDER_FRAME_COUNT],

    cache: Option<TargetCache>,
}

impl VulkanRenderer {
    pub fn new(ctx: Arc<VulkanContext>, config: crate::RendererConfig) -> Self {
        let msaa_samples = config.msaa_samples;
        let ssaa_factor = config.ssaa_factor;
        let analytic_aa_2d = config.analytic_aa_2d;
        let output_transform = match config.output_color_profile {
            crate::OutputColorProfile::Bt709Sdr => {
                include_str!("output_transform_bt709.wgsl")
            }
            crate::OutputColorProfile::Bt2020Pq | crate::OutputColorProfile::Bt2020Hlg => {
                panic!(
                    "{:?} requires a 10-bit HDR output path; the current renderer outputs RGBA8/NV12",
                    config.output_color_profile
                )
            }
        };
        let requested_sample_count = msaa_to_vk_sample_count(msaa_samples);
        let sample_count = get_max_usable_sample_count(&ctx, requested_sample_count);
        let (environment_map, environment_sampler) = create_studio_environment(&ctx);

        let pipelines = PipelineSet::new(&ctx, output_transform, sample_count);

        let descriptor_pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 96,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLED_IMAGE,
                descriptor_count: 96,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLER,
                descriptor_count: 8,
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
            max_sets: 48,
            flags: vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET,
            ..Default::default()
        };
        let descriptor_pool = unsafe {
            ctx.device
                .create_descriptor_pool(&descriptor_pool_info, None)
                .unwrap()
        };

        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);
        let texture_sampler = unsafe { ctx.device.create_sampler(&sampler_info, None).unwrap() };

        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(std::slice::from_ref(&pipelines.raster_texture_layout));
        let raster_texture_set =
            unsafe { ctx.device.allocate_descriptor_sets(&alloc_info).unwrap()[0] };

        let dummy_texture = Image::new(
            &ctx,
            1,
            1,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            vk::ImageAspectFlags::COLOR,
            vk::SampleCountFlags::TYPE_1,
        );
        let command_pool = unsafe {
            ctx.device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(ctx.queue_family_index)
                        .flags(vk::CommandPoolCreateFlags::TRANSIENT),
                    None,
                )
                .unwrap()
        };
        let command_buffer = unsafe {
            ctx.device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(command_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .unwrap()[0]
        };
        unsafe {
            ctx.device
                .begin_command_buffer(
                    command_buffer,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .unwrap();
            let barrier = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(dummy_texture.vk_image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            ctx.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
            ctx.device.end_command_buffer(command_buffer).unwrap();
            let submit_info =
                vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&command_buffer));
            ctx.device
                .queue_submit(ctx.queue, &[submit_info], vk::Fence::null())
                .unwrap();
            ctx.device.queue_wait_idle(ctx.queue).unwrap();
            ctx.device.destroy_command_pool(command_pool, None);
        }

        let dummy_image_infos: Vec<_> = (0..16)
            .map(|_| {
                vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(dummy_texture.view)
            })
            .collect();
        let images_write = vk::WriteDescriptorSet::default()
            .dst_set(raster_texture_set)
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .image_info(&dummy_image_infos);

        let sampler_image_info = [vk::DescriptorImageInfo::default().sampler(texture_sampler)];
        let sampler_write = vk::WriteDescriptorSet::default()
            .dst_set(raster_texture_set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::SAMPLER)
            .image_info(&sampler_image_info);
        unsafe {
            ctx.device
                .update_descriptor_sets(&[images_write, sampler_write], &[]);
        }

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
        let material_buffer_3d_stride = align_up(
            (std::mem::size_of::<MaterialData3D>() * MAX_SURFACE_MATERIALS) as u64,
            storage_alignment,
        );
        let primitive_buffer_stride = align_up(
            (std::mem::size_of::<GpuSdfPrimitive>() * 10_000) as u64,
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
        let tone_map_factor_stride = align_up(
            std::mem::size_of::<ToneMapConstants>() as u64,
            uniform_alignment,
        );
        let frame_count = RENDER_FRAME_COUNT as u64;

        let tone_map_factor_buffer = Buffer::new(
            &ctx,
            tone_map_factor_stride * frame_count,
            vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            gpu_allocator::MemoryLocation::CpuToGpu,
        );

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
        let material_buffer_3d = Buffer::new(
            &ctx,
            material_buffer_3d_stride * frame_count,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
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
            static_vertex_buffer_2d_size + vertex_staging_buffer_2d_stride * frame_count,
            vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            gpu_allocator::MemoryLocation::GpuOnly,
        );
        let index_buffer_2d = Buffer::new(
            &ctx,
            static_index_buffer_2d_size + index_staging_buffer_2d_stride * frame_count,
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

        Self {
            ctx,
            environment_map,
            environment_sampler,
            descriptor_pool,
            pipelines,
            vertex_buffer,
            index_buffer,
            camera_buffer,
            material_buffer_3d,
            buffer_3d,
            nv12_constants_buffer,
            raster_texture_set,
            active_textures: vec![dummy_texture],
            texture_sampler,
            vertex_buffer_2d,
            index_buffer_2d,
            vertex_staging_buffer_2d,
            index_staging_buffer_2d,
            instance_buffer_2d,
            camera_buffer_2d,
            tone_map_factor_buffer,
            vertex_buffer_stride,
            index_buffer_stride,
            camera_buffer_stride,
            material_buffer_3d_stride,
            primitive_buffer_stride,
            vertex_staging_buffer_2d_stride,
            index_staging_buffer_2d_stride,
            instance_buffer_2d_stride,
            camera_buffer_2d_stride,
            tone_map_factor_stride,
            mesh_upload_planner_2d: Mesh2DUploadPlanner::new(
                static_vertex_buffer_2d_size,
                static_index_buffer_2d_size,
            ),
            scene_preparer: ScenePreparer::default(),
            last_stats: RendererStats::default(),
            gpu_profiling: false,
            last_gpu_timings: None,
            bloom_enabled: false,
            analytic_aa_2d,
            frame_data,
            cache: None,
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

    pub fn set_bloom_enabled(&mut self, enabled: bool) {
        self.bloom_enabled = enabled;
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
        let prepared = self
            .scene_preparer
            .prepare(scene, scene_config, self.ssaa_factor);
        let objects_3d = prepared
            .sdf_primitives
            .iter()
            .map(|prepared| GpuSdfPrimitive::encode(prepared.primitive, prepared.material_index))
            .collect::<Vec<_>>();

        self.render(
            scene_config.output_width,
            scene_config.output_height,
            &prepared.camera_uniform,
            &prepared.camera_uniform_2d,
            &objects_3d,
            &prepared.surface_materials,
            &prepared.mesh_vertices,
            &prepared.mesh_indices,
            &prepared.mesh_draws_3d,
            &prepared.mesh_batches_2d,
            output,
            outputs,
            prepared.background_color,
        );
    }

    pub fn update_texture(&mut self, index: u32, width: u32, height: u32, data: &[u8]) {
        assert!(index < 16, "Texture index out of bounds");

        let byte_offset = (width * height * 4) as u64;
        let mut staging = Buffer::new(
            &self.ctx,
            byte_offset,
            vk::BufferUsageFlags::TRANSFER_SRC,
            gpu_allocator::MemoryLocation::CpuToGpu,
        );
        staging.write_bytes(0, data);

        let image = Image::new(
            &self.ctx,
            width,
            height,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
            vk::ImageAspectFlags::COLOR,
            vk::SampleCountFlags::TYPE_1,
        );

        let command_pool = unsafe {
            self.ctx
                .device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(self.ctx.queue_family_index)
                        .flags(vk::CommandPoolCreateFlags::TRANSIENT),
                    None,
                )
                .unwrap()
        };
        let command_buffer = unsafe {
            self.ctx
                .device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(command_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .unwrap()[0]
        };

        unsafe {
            self.ctx
                .device
                .begin_command_buffer(
                    command_buffer,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .unwrap();

            let mut barrier = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image.vk_image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);

            self.ctx.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );

            let region = vk::BufferImageCopy::default()
                .buffer_offset(0)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                });

            self.ctx.device.cmd_copy_buffer_to_image(
                command_buffer,
                staging.vk_buffer,
                image.vk_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );

            barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
            barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
            barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
            barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

            self.ctx.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );

            self.ctx.device.end_command_buffer(command_buffer).unwrap();

            let submit_info =
                vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&command_buffer));
            self.ctx
                .device
                .queue_submit(self.ctx.queue, &[submit_info], vk::Fence::null())
                .unwrap();
            self.ctx.device.queue_wait_idle(self.ctx.queue).unwrap();
            self.ctx.device.destroy_command_pool(command_pool, None);

            let image_info = [vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(image.view)];

            let write = vk::WriteDescriptorSet::default()
                .dst_set(self.raster_texture_set)
                .dst_binding(0)
                .dst_array_element(index)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .image_info(&image_info);

            self.ctx.device.update_descriptor_sets(&[write], &[]);
        }

        self.active_textures.push(image);
    }

    pub fn current_output_image_view(&self) -> Option<vk::ImageView> {
        let cache = self.cache.as_ref()?;
        if cache.current_frame == 0 {
            return None;
        }
        let frame_idx = (cache.current_frame - 1) % RENDER_FRAME_COUNT;
        Some(cache.render_targets[frame_idx].texture.view)
    }

    pub fn bind_texture_view(&mut self, index: u32, view: vk::ImageView) {
        assert!(index < 16, "Texture index out of bounds");
        let image_info = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(view)];
        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.raster_texture_set)
            .dst_binding(0)
            .dst_array_element(index)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .image_info(&image_info);
        unsafe {
            self.ctx.device.update_descriptor_sets(&[write], &[]);
        }
    }

    pub fn wait_idle(&self) {
        unsafe {
            let _ = self.ctx.device.device_wait_idle();
        }
    }

    fn render(
        &mut self,
        width: u32,
        height: u32,
        camera_uniform: &CameraUniform,
        camera_uniform_2d: &CameraUniform2D,
        objects_3d: &[GpuSdfPrimitive],
        surface_materials: &[SurfaceMaterial],
        mesh_vertices: &[Vertex],
        mesh_indices: &[u32],
        mesh_draws_3d: &[Mesh3DDraw],
        mesh_batches_2d: &[Mesh2DBatch],
        output: Option<&mut [u8]>,
        outputs: RenderOutputs,
        background_color: [f32; 4],
    ) {
        let align = 256;
        let unpadded_bytes_per_row = width * 4;
        let padded_bytes_per_row = (unpadded_bytes_per_row + align - 1) & !(align - 1);

        let needs_raster_gbuffer = mesh_draws_3d.iter().any(|draw| !draw.is_transparent());
        let needs_overlay_hdr = (needs_raster_gbuffer || !objects_3d.is_empty())
            && (mesh_draws_3d.iter().any(|draw| draw.is_transparent())
                || !mesh_batches_2d.is_empty());
        let cache_needs_update = self.cache.as_ref().map_or(true, |c| {
            c.width != width
                || c.height != height
                || (needs_raster_gbuffer && !c.has_raster_gbuffer)
                || (needs_overlay_hdr && !c.has_overlay_hdr)
        });

        if cache_needs_update {
            if let Some(mut old_cache) = self.cache.take() {
                unsafe {
                    self.ctx.device.device_wait_idle().unwrap();
                }
                old_cache.destroy(&self.ctx, self.descriptor_pool);
            }
            let resources = TargetCacheResources {
                ctx: &self.ctx,
                descriptor_pool: self.descriptor_pool,
                pipelines: &self.pipelines,
                msaa_samples: self.msaa_samples,
                ssaa_factor: self.ssaa_factor,
                environment_map: &self.environment_map,
                environment_sampler: self.environment_sampler,
                camera_buffer: &self.camera_buffer,
                material_buffer_3d: &self.material_buffer_3d,
                primitive_buffer: &self.buffer_3d,
                camera_buffer_2d: &self.camera_buffer_2d,
                nv12_constants_buffer: &self.nv12_constants_buffer,
                tone_map_factor_buffer: &self.tone_map_factor_buffer,
                camera_buffer_stride: self.camera_buffer_stride,
                material_buffer_3d_stride: self.material_buffer_3d_stride,
                primitive_buffer_stride: self.primitive_buffer_stride,
                camera_buffer_2d_stride: self.camera_buffer_2d_stride,
                tone_map_factor_stride: self.tone_map_factor_stride,
            };
            self.cache = Some(TargetCache::new(
                width,
                height,
                padded_bytes_per_row,
                needs_raster_gbuffer,
                needs_overlay_hdr,
                &resources,
            ));
        }

        let cache = self.cache.as_mut().unwrap();
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
        let mesh_2d_arenas = self.mesh_upload_planner_2d.frame_arenas(
            frame_idx as u64,
            self.vertex_staging_buffer_2d_stride,
            self.index_staging_buffer_2d_stride,
            self.instance_buffer_2d_stride,
        );
        let prepared_2d = self
            .mesh_upload_planner_2d
            .prepare(mesh_2d_arenas, mesh_batches_2d);
        let PreparedMesh2D {
            batches: prepared_mesh_batches_2d,
            uploads: geometry_uploads_2d,
            instances: instances_2d,
        } = match prepared_2d {
            Ok(prepared) => prepared,
            Err(PrepareMesh2DError::StaticArenaExhausted) => {
                unsafe {
                    self.ctx.device.device_wait_idle().unwrap();
                }
                self.mesh_upload_planner_2d.reset_static_arena();
                mesh_2d_arena_rebuilds = 1;
                self.mesh_upload_planner_2d
                    .prepare(mesh_2d_arenas, mesh_batches_2d)
                    .expect("active 2D scene exceeds a frame or persistent geometry arena")
            }
            Err(error) => panic!("2D frame preparation failed: {error:?}"),
        };
        let has_sdf = !objects_3d.is_empty();
        // Analytic AA applies only when every rastered 2D instance is a
        // filled rectangle: the frame then renders at output resolution with
        // one sample, and the tone-map downsample factor becomes one.
        // Bloom is excluded for now because its extract pass still derives
        // its sampling grid from the resolved image dimensions.
        let raster_2d_only = objects_3d.is_empty()
            && mesh_indices.is_empty()
            && !prepared_mesh_batches_2d.is_empty();
        let analytic_2d = raster_2d_only
            && self.analytic_aa_2d
            && !self.bloom_enabled
            && instances_2d
                .iter()
                .all(|instance| instance.aa_params[2] > 0.5);
        let frame_plan = if analytic_2d {
            FrameExecutionPlan::RasterToneMap
        } else {
            FrameExecutionPlan::build(
                has_sdf,
                !mesh_indices.is_empty() || !prepared_mesh_batches_2d.is_empty(),
                self.ssaa_factor,
            )
        };
        let raster_scale = if analytic_2d { 1 } else { self.ssaa_factor };
        let fused_video_downsample = frame_plan == FrameExecutionPlan::RasterDownsample
            && self.ssaa_factor == 2
            && !self.bloom_enabled
            && outputs.vulkan_video
            && !outputs.cpu_nv12
            && !outputs.cpu_yuv444p
            && !outputs.cpu_rgba;
        let runs_postprocess = frame_plan != FrameExecutionPlan::Empty && !fused_video_downsample;
        let raster_uses_depth = !mesh_indices.is_empty();
        let has_transparent_meshes = mesh_draws_3d.iter().any(|draw| draw.is_transparent());
        let has_opaque_meshes = mesh_draws_3d.iter().any(|draw| !draw.is_transparent());
        let uses_deferred_raster = has_opaque_meshes;
        let has_surface_overlay = (frame_plan.runs_sdf() || uses_deferred_raster)
            && (has_transparent_meshes || !prepared_mesh_batches_2d.is_empty());
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
        let raster_sample_count = msaa_to_vk_sample_count(self.msaa_samples);
        if frame_plan.runs_raster()
            && !analytic_2d
            && cache.msaa_texture.is_none()
            && raster_sample_count != vk::SampleCountFlags::TYPE_1
        {
            cache.msaa_texture = Some(Image::new(
                &self.ctx,
                width * self.ssaa_factor,
                height * self.ssaa_factor,
                vk::Format::R16G16B16A16_SFLOAT,
                vk::ImageUsageFlags::COLOR_ATTACHMENT,
                vk::ImageAspectFlags::COLOR,
                raster_sample_count,
            ));
            cache.msaa_texture_state = TrackedImageState::UNDEFINED;
        }

        self.last_stats = RendererStats {
            mesh_3d_opaque_draw_calls: mesh_draws_3d
                .iter()
                .filter(|draw| !draw.is_transparent())
                .count() as u32,
            mesh_3d_transparent_draw_calls: mesh_draws_3d
                .iter()
                .filter(|draw| draw.is_transparent())
                .count() as u32
                * 2,
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
            mesh_2d_analytic_aa: analytic_2d as u32,
            sdf_dispatches: frame_plan.runs_sdf() as u32,
            surface_lighting_dispatches: (frame_plan.runs_sdf() || uses_deferred_raster) as u32,
            raster_passes: frame_plan.runs_raster() as u32 + has_transparent_meshes as u32 * 2,
            depth_attachment_raster_passes: raster_uses_depth as u32
                * (1 + has_transparent_meshes as u32),
            tone_map_dispatches: runs_postprocess as u32,
            bloom_dispatches: if self.bloom_enabled && frame_plan != FrameExecutionPlan::Empty {
                3
            } else {
                0
            },
            downsample_dispatches: (frame_plan == FrameExecutionPlan::RasterDownsample
                && !fused_video_downsample) as u32,
            fused_video_downsample_dispatches: fused_video_downsample as u32,
            surface_resolve_dispatches: (frame_plan.runs_sdf() || uses_deferred_raster) as u32,
            surface_composite_dispatches: (frame_plan.runs_sdf() || uses_deferred_raster) as u32,
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
        let material_buffer_3d_offset = self.material_buffer_3d_stride * frame_idx as u64;
        let primitive_buffer_offset = self.primitive_buffer_stride * frame_idx as u64;
        let vertex_staging_buffer_2d_offset =
            self.vertex_staging_buffer_2d_stride * frame_idx as u64;
        let index_staging_buffer_2d_offset = self.index_staging_buffer_2d_stride * frame_idx as u64;
        let instance_buffer_2d_offset = self.instance_buffer_2d_stride * frame_idx as u64;
        let camera_buffer_2d_offset = self.camera_buffer_2d_stride * frame_idx as u64;
        let compute_dynamic_offsets = [camera_buffer_offset as u32, primitive_buffer_offset as u32];
        let surface_dynamic_offsets = [
            camera_buffer_offset as u32,
            material_buffer_3d_offset as u32,
        ];
        let raster_dynamic_offsets = [
            camera_buffer_offset as u32,
            material_buffer_3d_offset as u32,
        ];
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
            self.tone_map_factor_buffer.write_bytes(
                frame_idx as u64 * self.tone_map_factor_stride,
                bytemuck::bytes_of(&ToneMapConstants {
                    factor: raster_scale,
                    _padding: [0; 3],
                }),
            );
            if !surface_materials.is_empty() {
                assert!(
                    surface_materials.len() <= MAX_SURFACE_MATERIALS,
                    "3D material count exceeds {MAX_SURFACE_MATERIALS}"
                );
                let materials: Vec<MaterialData3D> = surface_materials
                    .iter()
                    .copied()
                    .map(MaterialData3D::from)
                    .collect();
                self.material_buffer_3d
                    .write_bytes(material_buffer_3d_offset, bytemuck::cast_slice(&materials));
            }
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
            let recorder =
                CommandRecorder::new(&self.ctx.device, fd.command_buffer, &self.pipelines);
            recorder.record_geometry_uploads_2d(
                &geometry_uploads_2d,
                GeometryUploadBuffers2D {
                    vertex_staging: self.vertex_staging_buffer_2d.vk_buffer,
                    vertex_staging_base: vertex_staging_buffer_2d_offset,
                    index_staging: self.index_staging_buffer_2d.vk_buffer,
                    index_staging_base: index_staging_buffer_2d_offset,
                    vertex_device: self.vertex_buffer_2d.vk_buffer,
                    index_device: self.index_buffer_2d.vk_buffer,
                },
            );
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
                recorder.record_empty_frame(targets, background_color);
            }

            if frame_plan.runs_sdf() {
                recorder.record_sdf(targets, &compute_dynamic_offsets, width, height);
            }
            write_gpu_timestamp(
                &self.ctx.device,
                fd.command_buffer,
                fd.query_pool,
                2,
                gpu_profiling,
            );

            if uses_deferred_raster {
                for (image, state) in [
                    (
                        cache.raster_normal_depth.vk_image,
                        &mut cache.raster_normal_depth_state,
                    ),
                    (cache.raster_albedo.vk_image, &mut cache.raster_albedo_state),
                    (
                        cache.raster_material_id.vk_image,
                        &mut cache.raster_material_id_state,
                    ),
                ] {
                    transition_image(
                        &self.ctx.device,
                        fd.command_buffer,
                        image,
                        vk::ImageAspectFlags::COLOR,
                        state,
                        color_attachment_state,
                    );
                }
                let depth_texture = cache
                    .msaa_depth_texture
                    .as_ref()
                    .expect("deferred raster requires a depth attachment");
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
                let raster_extent = vk::Extent2D {
                    width: width * self.ssaa_factor,
                    height: height * self.ssaa_factor,
                };
                let gbuffer_attachments = [
                    vk::RenderingAttachmentInfo::default()
                        .image_view(cache.raster_normal_depth.view)
                        .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .load_op(vk::AttachmentLoadOp::CLEAR)
                        .store_op(vk::AttachmentStoreOp::STORE)
                        .clear_value(vk::ClearValue {
                            color: vk::ClearColorValue { float32: [0.0; 4] },
                        }),
                    vk::RenderingAttachmentInfo::default()
                        .image_view(cache.raster_albedo.view)
                        .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .load_op(vk::AttachmentLoadOp::CLEAR)
                        .store_op(vk::AttachmentStoreOp::STORE)
                        .clear_value(vk::ClearValue {
                            color: vk::ClearColorValue { float32: [0.0; 4] },
                        }),
                    vk::RenderingAttachmentInfo::default()
                        .image_view(cache.raster_material_id.view)
                        .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .load_op(vk::AttachmentLoadOp::CLEAR)
                        .store_op(vk::AttachmentStoreOp::STORE)
                        .clear_value(vk::ClearValue {
                            color: vk::ClearColorValue { uint32: [0; 4] },
                        }),
                ];
                let depth_attachment = vk::RenderingAttachmentInfo::default()
                    .image_view(depth_texture.view)
                    .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(if has_transparent_meshes {
                        vk::AttachmentStoreOp::STORE
                    } else {
                        vk::AttachmentStoreOp::DONT_CARE
                    })
                    .clear_value(vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue {
                            depth: 1.0,
                            stencil: 0,
                        },
                    });
                self.ctx.device.cmd_begin_rendering(
                    fd.command_buffer,
                    &vk::RenderingInfo::default()
                        .render_area(vk::Rect2D {
                            offset: vk::Offset2D { x: 0, y: 0 },
                            extent: raster_extent,
                        })
                        .layer_count(1)
                        .color_attachments(&gbuffer_attachments)
                        .depth_attachment(&depth_attachment),
                );
                let (vp_x, vp_y, vp_w, vp_h) = if camera_uniform.has_clip != 0 {
                    (
                        camera_uniform.clip_x * camera_uniform.raster_scale as f32,
                        camera_uniform.clip_y * camera_uniform.raster_scale as f32,
                        camera_uniform.clip_w * camera_uniform.raster_scale as f32,
                        camera_uniform.clip_h * camera_uniform.raster_scale as f32,
                    )
                } else {
                    (
                        0.0,
                        0.0,
                        raster_extent.width as f32,
                        raster_extent.height as f32,
                    )
                };
                self.ctx.device.cmd_set_viewport(
                    fd.command_buffer,
                    0,
                    &[vk::Viewport {
                        x: vp_x,
                        y: vp_y,
                        width: vp_w,
                        height: vp_h,
                        min_depth: 0.0,
                        max_depth: 1.0,
                    }],
                );
                self.ctx.device.cmd_set_scissor(
                    fd.command_buffer,
                    0,
                    &[vk::Rect2D {
                        offset: vk::Offset2D {
                            x: vp_x as i32,
                            y: vp_y as i32,
                        },
                        extent: vk::Extent2D {
                            width: vp_w as u32,
                            height: vp_h as u32,
                        },
                    }],
                );
                recorder.record_meshes_3d(
                    Mesh3DPass::Opaque,
                    Mesh3DBindings {
                        draws: mesh_draws_3d,
                        descriptor_set: targets.raster_descriptor_set,
                        dynamic_offsets: &raster_dynamic_offsets,
                        vertex_buffer: self.vertex_buffer.vk_buffer,
                        vertex_offset: vertex_buffer_offset,
                        index_buffer: self.index_buffer.vk_buffer,
                        index_offset: index_buffer_offset,
                    },
                );
                self.ctx.device.cmd_end_rendering(fd.command_buffer);

                for (image, state) in [
                    (
                        cache.raster_normal_depth.vk_image,
                        &mut cache.raster_normal_depth_state,
                    ),
                    (cache.raster_albedo.vk_image, &mut cache.raster_albedo_state),
                    (
                        cache.raster_material_id.vk_image,
                        &mut cache.raster_material_id_state,
                    ),
                ] {
                    transition_image(
                        &self.ctx.device,
                        fd.command_buffer,
                        image,
                        vk::ImageAspectFlags::COLOR,
                        state,
                        compute_read_state,
                    );
                }
                if !frame_plan.runs_sdf() {
                    for (image, state) in [
                        (
                            targets.sdf_normal_coverage.vk_image,
                            &mut targets.sdf_normal_coverage_state,
                        ),
                        (
                            targets.sdf_material_id.vk_image,
                            &mut targets.sdf_material_id_state,
                        ),
                        (targets.sdf_depth.vk_image, &mut targets.sdf_depth_state),
                    ] {
                        transition_image(
                            &self.ctx.device,
                            fd.command_buffer,
                            image,
                            vk::ImageAspectFlags::COLOR,
                            state,
                            compute_read_state,
                        );
                    }
                }
                record_surface_compute(
                    &self.ctx.device,
                    fd.command_buffer,
                    targets,
                    SurfaceComputePipelines {
                        resolve: self.pipelines.surface_resolve_pipeline,
                        resolve_layout: self.pipelines.surface_resolve_pipeline_layout,
                        lighting: self.pipelines.surface_lighting_pipeline,
                        lighting_layout: self.pipelines.surface_lighting_pipeline_layout,
                    },
                    &surface_dynamic_offsets,
                    raster_extent,
                );

                if has_surface_overlay {
                    if has_transparent_meshes {
                        if frame_plan.runs_sdf() {
                            transition_image(
                                &self.ctx.device,
                                fd.command_buffer,
                                targets.sdf_depth.vk_image,
                                vk::ImageAspectFlags::COLOR,
                                &mut targets.sdf_depth_state,
                                TrackedImageState {
                                    layout: vk::ImageLayout::GENERAL,
                                    stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                                    access: vk::AccessFlags2::SHADER_READ,
                                },
                            );
                        }
                        transition_image(
                            &self.ctx.device,
                            fd.command_buffer,
                            targets.surface_hdr.vk_image,
                            vk::ImageAspectFlags::COLOR,
                            &mut targets.surface_hdr_state,
                            TrackedImageState {
                                layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                                stage: vk::PipelineStageFlags2::COPY,
                                access: vk::AccessFlags2::TRANSFER_READ,
                            },
                        );
                        transition_image(
                            &self.ctx.device,
                            fd.command_buffer,
                            targets.scene_color.vk_image,
                            vk::ImageAspectFlags::COLOR,
                            &mut targets.scene_color_state,
                            TrackedImageState {
                                layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                                stage: vk::PipelineStageFlags2::COPY,
                                access: vk::AccessFlags2::TRANSFER_WRITE,
                            },
                        );
                        self.ctx.device.cmd_copy_image(
                            fd.command_buffer,
                            targets.surface_hdr.vk_image,
                            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                            targets.scene_color.vk_image,
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            &[vk::ImageCopy::default()
                                .src_subresource(vk::ImageSubresourceLayers {
                                    aspect_mask: vk::ImageAspectFlags::COLOR,
                                    mip_level: 0,
                                    base_array_layer: 0,
                                    layer_count: 1,
                                })
                                .dst_subresource(vk::ImageSubresourceLayers {
                                    aspect_mask: vk::ImageAspectFlags::COLOR,
                                    mip_level: 0,
                                    base_array_layer: 0,
                                    layer_count: 1,
                                })
                                .extent(vk::Extent3D {
                                    width: raster_extent.width,
                                    height: raster_extent.height,
                                    depth: 1,
                                })],
                        );
                        transition_image(
                            &self.ctx.device,
                            fd.command_buffer,
                            targets.scene_color.vk_image,
                            vk::ImageAspectFlags::COLOR,
                            &mut targets.scene_color_state,
                            TrackedImageState {
                                layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                                stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                                access: vk::AccessFlags2::SHADER_READ,
                            },
                        );

                        transition_image(
                            &self.ctx.device,
                            fd.command_buffer,
                            targets.transparent_back_depth.vk_image,
                            vk::ImageAspectFlags::COLOR,
                            &mut targets.transparent_back_depth_state,
                            color_attachment_state,
                        );
                        let thickness_attachment = vk::RenderingAttachmentInfo::default()
                            .image_view(targets.transparent_back_depth.view)
                            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                            .load_op(vk::AttachmentLoadOp::CLEAR)
                            .store_op(vk::AttachmentStoreOp::STORE)
                            .clear_value(vk::ClearValue {
                                color: vk::ClearColorValue { float32: [0.0; 4] },
                            });
                        self.ctx.device.cmd_begin_rendering(
                            fd.command_buffer,
                            &vk::RenderingInfo::default()
                                .render_area(vk::Rect2D {
                                    offset: vk::Offset2D { x: 0, y: 0 },
                                    extent: raster_extent,
                                })
                                .layer_count(1)
                                .color_attachments(std::slice::from_ref(&thickness_attachment)),
                        );
                        recorder.record_meshes_3d(
                            Mesh3DPass::TransparentDepth,
                            Mesh3DBindings {
                                draws: mesh_draws_3d,
                                descriptor_set: targets.raster_descriptor_set,
                                dynamic_offsets: &raster_dynamic_offsets,
                                vertex_buffer: self.vertex_buffer.vk_buffer,
                                vertex_offset: vertex_buffer_offset,
                                index_buffer: self.index_buffer.vk_buffer,
                                index_offset: index_buffer_offset,
                            },
                        );
                        self.ctx.device.cmd_end_rendering(fd.command_buffer);
                        transition_image(
                            &self.ctx.device,
                            fd.command_buffer,
                            targets.transparent_back_depth.vk_image,
                            vk::ImageAspectFlags::COLOR,
                            &mut targets.transparent_back_depth_state,
                            TrackedImageState {
                                layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                                stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                                access: vk::AccessFlags2::SHADER_READ,
                            },
                        );
                    }

                    transition_image(
                        &self.ctx.device,
                        fd.command_buffer,
                        targets.overlay_hdr.vk_image,
                        vk::ImageAspectFlags::COLOR,
                        &mut targets.overlay_hdr_state,
                        color_attachment_state,
                    );
                    let mut overlay_attachment = vk::RenderingAttachmentInfo::default()
                        .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .load_op(vk::AttachmentLoadOp::CLEAR)
                        .store_op(vk::AttachmentStoreOp::STORE)
                        .clear_value(vk::ClearValue {
                            color: vk::ClearColorValue { float32: [0.0; 4] },
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
                        overlay_attachment = overlay_attachment
                            .image_view(msaa_texture.view)
                            .store_op(vk::AttachmentStoreOp::DONT_CARE)
                            .resolve_mode(vk::ResolveModeFlags::AVERAGE)
                            .resolve_image_view(targets.overlay_hdr.view)
                            .resolve_image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
                    } else {
                        overlay_attachment =
                            overlay_attachment.image_view(targets.overlay_hdr.view);
                    }
                    let overlay_depth_attachment = vk::RenderingAttachmentInfo::default()
                        .image_view(
                            cache
                                .msaa_depth_texture
                                .as_ref()
                                .expect("deferred overlay requires a depth attachment")
                                .view,
                        )
                        .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                        .load_op(if has_transparent_meshes {
                            vk::AttachmentLoadOp::LOAD
                        } else {
                            vk::AttachmentLoadOp::DONT_CARE
                        })
                        .store_op(vk::AttachmentStoreOp::DONT_CARE);
                    self.ctx.device.cmd_begin_rendering(
                        fd.command_buffer,
                        &vk::RenderingInfo::default()
                            .render_area(vk::Rect2D {
                                offset: vk::Offset2D { x: 0, y: 0 },
                                extent: raster_extent,
                            })
                            .layer_count(1)
                            .color_attachments(std::slice::from_ref(&overlay_attachment))
                            .depth_attachment(&overlay_depth_attachment),
                    );
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
                    if has_transparent_meshes {
                        recorder.record_meshes_3d(
                            Mesh3DPass::TransparentColor,
                            Mesh3DBindings {
                                draws: mesh_draws_3d,
                                descriptor_set: targets.raster_descriptor_set,
                                dynamic_offsets: &raster_dynamic_offsets,
                                vertex_buffer: self.vertex_buffer.vk_buffer,
                                vertex_offset: vertex_buffer_offset,
                                index_buffer: self.index_buffer.vk_buffer,
                                index_offset: index_buffer_offset,
                            },
                        );
                    }
                    recorder.record_meshes_2d(
                        Mesh2DPass::Depth,
                        Mesh2DBindings {
                            batches: &prepared_mesh_batches_2d,
                            camera_descriptor_set: cache.raster_descriptor_set_2d,
                            camera_dynamic_offsets: &raster_2d_dynamic_offsets,
                            texture_descriptor_set: self.raster_texture_set,
                            vertex_buffer: self.vertex_buffer_2d.vk_buffer,
                            index_buffer: self.index_buffer_2d.vk_buffer,
                            instance_buffer: self.instance_buffer_2d.vk_buffer,
                            instance_offset: instance_buffer_2d_offset,
                        },
                    );
                    self.ctx.device.cmd_end_rendering(fd.command_buffer);
                }
            }

            if frame_plan.runs_sdf() && !uses_deferred_raster {
                for (image, state) in [
                    (
                        cache.raster_normal_depth.vk_image,
                        &mut cache.raster_normal_depth_state,
                    ),
                    (cache.raster_albedo.vk_image, &mut cache.raster_albedo_state),
                    (
                        cache.raster_material_id.vk_image,
                        &mut cache.raster_material_id_state,
                    ),
                ] {
                    transition_image(
                        &self.ctx.device,
                        fd.command_buffer,
                        image,
                        vk::ImageAspectFlags::COLOR,
                        state,
                        compute_read_state,
                    );
                }
                record_surface_compute(
                    &self.ctx.device,
                    fd.command_buffer,
                    targets,
                    SurfaceComputePipelines {
                        resolve: self.pipelines.surface_resolve_pipeline,
                        resolve_layout: self.pipelines.surface_resolve_pipeline_layout,
                        lighting: self.pipelines.surface_lighting_pipeline,
                        lighting_layout: self.pipelines.surface_lighting_pipeline_layout,
                    },
                    &surface_dynamic_offsets,
                    vk::Extent2D {
                        width: width * self.ssaa_factor,
                        height: height * self.ssaa_factor,
                    },
                );
                if has_transparent_meshes {
                    transition_image(
                        &self.ctx.device,
                        fd.command_buffer,
                        targets.sdf_depth.vk_image,
                        vk::ImageAspectFlags::COLOR,
                        &mut targets.sdf_depth_state,
                        TrackedImageState {
                            layout: vk::ImageLayout::GENERAL,
                            stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                            access: vk::AccessFlags2::SHADER_READ,
                        },
                    );
                }
            }

            if frame_plan.runs_raster() && !uses_deferred_raster {
                let (target_image, target_view, target_state) =
                    if frame_plan == FrameExecutionPlan::SdfRasterComposite {
                        (
                            targets.overlay_hdr.vk_image,
                            targets.overlay_hdr.view,
                            &mut targets.overlay_hdr_state,
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
                            float32: background_color,
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
                        .store_op(vk::AttachmentStoreOp::STORE)
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
                        .store_op(if has_transparent_meshes {
                            vk::AttachmentStoreOp::STORE
                        } else {
                            vk::AttachmentStoreOp::DONT_CARE
                        })
                        .clear_value(vk::ClearValue {
                            depth_stencil: vk::ClearDepthStencilValue {
                                depth: 1.0,
                                stencil: 0,
                            },
                        })
                });
                let color_attachments = [color_attachment];
                let raster_extent = vk::Extent2D {
                    width: width * raster_scale,
                    height: height * raster_scale,
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

                let (vp_x, vp_y, vp_w, vp_h) = if camera_uniform.has_clip != 0 {
                    (
                        camera_uniform.clip_x * camera_uniform.raster_scale as f32,
                        camera_uniform.clip_y * camera_uniform.raster_scale as f32,
                        camera_uniform.clip_w * camera_uniform.raster_scale as f32,
                        camera_uniform.clip_h * camera_uniform.raster_scale as f32,
                    )
                } else {
                    (
                        0.0,
                        0.0,
                        raster_extent.width as f32,
                        raster_extent.height as f32,
                    )
                };
                self.ctx.device.cmd_set_viewport(
                    fd.command_buffer,
                    0,
                    &[vk::Viewport {
                        x: vp_x,
                        y: vp_y,
                        width: vp_w,
                        height: vp_h,
                        min_depth: 0.0,
                        max_depth: 1.0,
                    }],
                );
                self.ctx.device.cmd_set_scissor(
                    fd.command_buffer,
                    0,
                    &[vk::Rect2D {
                        offset: vk::Offset2D {
                            x: vp_x as i32,
                            y: vp_y as i32,
                        },
                        extent: vk::Extent2D {
                            width: vp_w as u32,
                            height: vp_h as u32,
                        },
                    }],
                );

                if !has_transparent_meshes && !prepared_mesh_batches_2d.is_empty() {
                    recorder.record_meshes_2d(
                        if analytic_2d {
                            Mesh2DPass::Analytic
                        } else if raster_uses_depth {
                            Mesh2DPass::Depth
                        } else {
                            Mesh2DPass::Depthless
                        },
                        Mesh2DBindings {
                            batches: &prepared_mesh_batches_2d,
                            camera_descriptor_set: cache.raster_descriptor_set_2d,
                            camera_dynamic_offsets: &raster_2d_dynamic_offsets,
                            texture_descriptor_set: self.raster_texture_set,
                            vertex_buffer: self.vertex_buffer_2d.vk_buffer,
                            index_buffer: self.index_buffer_2d.vk_buffer,
                            instance_buffer: self.instance_buffer_2d.vk_buffer,
                            instance_offset: instance_buffer_2d_offset,
                        },
                    );
                }
                self.ctx.device.cmd_end_rendering(fd.command_buffer);

                if has_transparent_meshes {
                    let (scene_source_image, scene_source_state) = if frame_plan.runs_sdf() {
                        (targets.surface_hdr.vk_image, &mut targets.surface_hdr_state)
                    } else {
                        (target_image, &mut *target_state)
                    };
                    let transfer_source_state = TrackedImageState {
                        layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        stage: vk::PipelineStageFlags2::COPY,
                        access: vk::AccessFlags2::TRANSFER_READ,
                    };
                    let transfer_destination_state = TrackedImageState {
                        layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        stage: vk::PipelineStageFlags2::COPY,
                        access: vk::AccessFlags2::TRANSFER_WRITE,
                    };
                    transition_image(
                        &self.ctx.device,
                        fd.command_buffer,
                        scene_source_image,
                        vk::ImageAspectFlags::COLOR,
                        scene_source_state,
                        transfer_source_state,
                    );
                    transition_image(
                        &self.ctx.device,
                        fd.command_buffer,
                        targets.scene_color.vk_image,
                        vk::ImageAspectFlags::COLOR,
                        &mut targets.scene_color_state,
                        transfer_destination_state,
                    );
                    let copy_region = vk::ImageCopy::default()
                        .src_subresource(vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .dst_subresource(vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .extent(vk::Extent3D {
                            width: raster_extent.width,
                            height: raster_extent.height,
                            depth: 1,
                        });
                    self.ctx.device.cmd_copy_image(
                        fd.command_buffer,
                        scene_source_image,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        targets.scene_color.vk_image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        std::slice::from_ref(&copy_region),
                    );
                    transition_image(
                        &self.ctx.device,
                        fd.command_buffer,
                        scene_source_image,
                        vk::ImageAspectFlags::COLOR,
                        scene_source_state,
                        if frame_plan.runs_sdf() {
                            compute_read_state
                        } else {
                            color_attachment_state
                        },
                    );
                    transition_image(
                        &self.ctx.device,
                        fd.command_buffer,
                        targets.scene_color.vk_image,
                        vk::ImageAspectFlags::COLOR,
                        &mut targets.scene_color_state,
                        TrackedImageState {
                            layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                            stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                            access: vk::AccessFlags2::SHADER_READ,
                        },
                    );

                    transition_image(
                        &self.ctx.device,
                        fd.command_buffer,
                        targets.transparent_back_depth.vk_image,
                        vk::ImageAspectFlags::COLOR,
                        &mut targets.transparent_back_depth_state,
                        color_attachment_state,
                    );
                    let thickness_attachment = vk::RenderingAttachmentInfo::default()
                        .image_view(targets.transparent_back_depth.view)
                        .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .load_op(vk::AttachmentLoadOp::CLEAR)
                        .store_op(vk::AttachmentStoreOp::STORE)
                        .clear_value(vk::ClearValue {
                            color: vk::ClearColorValue { float32: [0.0; 4] },
                        });
                    let thickness_attachments = [thickness_attachment];
                    let thickness_rendering_info = vk::RenderingInfo::default()
                        .render_area(vk::Rect2D {
                            offset: vk::Offset2D { x: 0, y: 0 },
                            extent: raster_extent,
                        })
                        .layer_count(1)
                        .color_attachments(&thickness_attachments);
                    self.ctx
                        .device
                        .cmd_begin_rendering(fd.command_buffer, &thickness_rendering_info);
                    recorder.record_meshes_3d(
                        Mesh3DPass::TransparentDepth,
                        Mesh3DBindings {
                            draws: mesh_draws_3d,
                            descriptor_set: targets.raster_descriptor_set,
                            dynamic_offsets: &raster_dynamic_offsets,
                            vertex_buffer: self.vertex_buffer.vk_buffer,
                            vertex_offset: vertex_buffer_offset,
                            index_buffer: self.index_buffer.vk_buffer,
                            index_offset: index_buffer_offset,
                        },
                    );
                    self.ctx.device.cmd_end_rendering(fd.command_buffer);
                    transition_image(
                        &self.ctx.device,
                        fd.command_buffer,
                        targets.transparent_back_depth.vk_image,
                        vk::ImageAspectFlags::COLOR,
                        &mut targets.transparent_back_depth_state,
                        TrackedImageState {
                            layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                            stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                            access: vk::AccessFlags2::SHADER_READ,
                        },
                    );

                    let mut transparent_color_attachment = vk::RenderingAttachmentInfo::default()
                        .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .load_op(vk::AttachmentLoadOp::LOAD)
                        .store_op(vk::AttachmentStoreOp::STORE);
                    if let Some(msaa_texture) = &cache.msaa_texture {
                        transparent_color_attachment = transparent_color_attachment
                            .image_view(msaa_texture.view)
                            .store_op(vk::AttachmentStoreOp::DONT_CARE)
                            .resolve_mode(vk::ResolveModeFlags::AVERAGE)
                            .resolve_image_view(target_view)
                            .resolve_image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
                    } else {
                        transparent_color_attachment =
                            transparent_color_attachment.image_view(target_view);
                    }
                    let transparent_depth_attachment =
                        cache.msaa_depth_texture.as_ref().map(|texture| {
                            vk::RenderingAttachmentInfo::default()
                                .image_view(texture.view)
                                .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                                .load_op(vk::AttachmentLoadOp::LOAD)
                                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                        });
                    let transparent_color_attachments = [transparent_color_attachment];
                    let mut transparent_rendering_info = vk::RenderingInfo::default()
                        .render_area(vk::Rect2D {
                            offset: vk::Offset2D { x: 0, y: 0 },
                            extent: raster_extent,
                        })
                        .layer_count(1)
                        .color_attachments(&transparent_color_attachments);
                    if raster_uses_depth {
                        transparent_rendering_info = transparent_rendering_info.depth_attachment(
                            transparent_depth_attachment
                                .as_ref()
                                .expect("transparent 3D raster requires a depth attachment"),
                        );
                    }
                    self.ctx
                        .device
                        .cmd_begin_rendering(fd.command_buffer, &transparent_rendering_info);

                    recorder.record_meshes_3d(
                        Mesh3DPass::TransparentColor,
                        Mesh3DBindings {
                            draws: mesh_draws_3d,
                            descriptor_set: targets.raster_descriptor_set,
                            dynamic_offsets: &raster_dynamic_offsets,
                            vertex_buffer: self.vertex_buffer.vk_buffer,
                            vertex_offset: vertex_buffer_offset,
                            index_buffer: self.index_buffer.vk_buffer,
                            index_offset: index_buffer_offset,
                        },
                    );

                    recorder.record_meshes_2d(
                        Mesh2DPass::Depth,
                        Mesh2DBindings {
                            batches: &prepared_mesh_batches_2d,
                            camera_descriptor_set: cache.raster_descriptor_set_2d,
                            camera_dynamic_offsets: &raster_2d_dynamic_offsets,
                            texture_descriptor_set: self.raster_texture_set,
                            vertex_buffer: self.vertex_buffer_2d.vk_buffer,
                            index_buffer: self.index_buffer_2d.vk_buffer,
                            instance_buffer: self.instance_buffer_2d.vk_buffer,
                            instance_offset: instance_buffer_2d_offset,
                        },
                    );
                    self.ctx.device.cmd_end_rendering(fd.command_buffer);
                }
            }

            if frame_plan.runs_sdf() || uses_deferred_raster {
                recorder.record_surface_composite(
                    targets,
                    has_surface_overlay,
                    width,
                    height,
                    self.ssaa_factor,
                );
            }
            write_gpu_timestamp(
                &self.ctx.device,
                fd.command_buffer,
                fd.query_pool,
                3,
                gpu_profiling,
            );

            if frame_plan != FrameExecutionPlan::Empty {
                recorder.record_bloom(targets, self.bloom_enabled);
            }

            if runs_postprocess {
                recorder.record_tone_map(targets, width, height);
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
            let video_output = outputs.vulkan_video.then(|| {
                let slot = &cache.video_nv12_slots[video_frame_idx];
                VideoOutputPass {
                    image: slot.image.vk_image,
                    descriptor_set: slot.descriptor_set,
                    current_layout: slot.layout,
                }
            });
            recorder.record_outputs(
                targets,
                OutputPasses {
                    width,
                    height,
                    fused_video_downsample,
                    cpu_nv12_descriptor_set: outputs
                        .cpu_nv12
                        .then_some(cache.nv12_descriptor_sets[frame_idx]),
                    cpu_yuv444p_descriptor_set: outputs
                        .cpu_yuv444p
                        .then_some(cache.yuv444p_descriptor_sets[frame_idx]),
                    video: video_output,
                    rgba_buffer: outputs
                        .cpu_rgba
                        .then_some(cache.output_buffers[frame_idx].vk_buffer),
                    rgba_padded_bytes_per_row: cache.padded_bytes_per_row,
                },
            );
            if outputs.vulkan_video {
                cache.video_nv12_slots[video_frame_idx].layout = vk::ImageLayout::GENERAL;
            }
            write_gpu_timestamp(
                &self.ctx.device,
                fd.command_buffer,
                fd.query_pool,
                5,
                gpu_profiling,
            );

            if !has_compute_output && !outputs.cpu_rgba {
                transition_image(
                    &self.ctx.device,
                    fd.command_buffer,
                    targets.texture.vk_image,
                    vk::ImageAspectFlags::COLOR,
                    &mut targets.texture_state,
                    TrackedImageState {
                        layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                        access: vk::AccessFlags2::SHADER_READ,
                    },
                );
            }

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
        if let Some(cache) = self.cache.as_ref() {
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
        if let Some(cache) = self.cache.as_ref() {
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

    pub fn get_vulkan_video_frame(&mut self) -> Option<VulkanVideoFrame> {
        if let Some(cache) = self.cache.as_mut() {
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

    pub fn get_rgba_bytes(&mut self) -> Option<&[u8]> {
        if let Some(cache) = self.cache.as_mut() {
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
    use nalgebra::{Point3, Vector3};

    use super::{
        FrameExecutionPlan, GpuSdfPrimitive, MaterialData3D, timestamp_delta, video_timeline_values,
    };
    use crate::mobjects::mesh_3d::{
        AlphaMode3D, SphericalPatchMaterial, SurfaceMaterial, Transmission3D,
    };
    use crate::mobjects::object_3d::SdfPrimitive;

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
            FrameExecutionPlan::RasterToneMap
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
    fn sdf_primitives_encode_to_the_shader_abi() {
        assert_eq!(std::mem::size_of::<GpuSdfPrimitive>(), 64);

        let primitives = [
            SdfPrimitive::Sphere {
                center: Point3::new(1.0, 2.0, 3.0),
                radius: 4.0,
            },
            SdfPrimitive::Capsule {
                start: Point3::origin(),
                end: Point3::new(1.0, 2.0, 3.0),
                radius: 0.5,
            },
            SdfPrimitive::Arrow {
                start: Point3::origin(),
                end: Point3::new(1.0, 0.0, 0.0),
                shaft_radius: 0.1,
                head_radius: 0.2,
                head_length: 0.3,
            },
            SdfPrimitive::OrientedBox {
                center: Point3::new(1.0, 2.0, 3.0),
                half_extents: Vector3::new(4.0, 5.0, 6.0),
                x_axis: Vector3::new(1.0, 0.0, 0.0),
                y_axis: Vector3::new(0.0, 1.0, 0.0),
            },
            SdfPrimitive::QuadraticBezier {
                start: Point3::origin(),
                control: Point3::new(1.0, 2.0, 3.0),
                end: Point3::new(4.0, 5.0, 6.0),
                radius: 0.25,
            },
        ];

        for (shape_type, primitive) in primitives.into_iter().enumerate() {
            let encoded = GpuSdfPrimitive::encode(primitive, 17);
            assert_eq!(encoded.material_index, 17);
            assert_eq!(encoded.shape_type, shape_type as u32);
            assert_eq!(encoded.padding, [0; 2]);
        }

        let encoded = GpuSdfPrimitive::encode(primitives[3], 0);
        assert_eq!(
            encoded.params,
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,]
        );
    }

    #[test]
    fn spherical_patch_material_has_a_stable_gpu_layout() {
        let material = SurfaceMaterial {
            alpha_mode: AlphaMode3D::Blend(Transmission3D {
                opacity: 0.1,
                fresnel_opacity: 0.2,
                absorption: [0.3, 0.4, 0.5],
                ior: 1.7,
                backface_opacity_scale: 0.8,
            }),
            spherical_patch: Some(SphericalPatchMaterial {
                directions: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                color: [0.1, 0.2, 0.3, 0.4],
                edge_color: [0.5, 0.6, 0.7, 0.8],
                edge_width_pixels: 2.25,
            }),
            ..Default::default()
        };
        let gpu = MaterialData3D::from(material);

        assert_eq!(std::mem::size_of::<MaterialData3D>(), 14 * 16);
        assert_eq!(gpu.patch_corner_0, [1.0, 2.0, 3.0, 0.0]);
        assert_eq!(gpu.patch_corner_2, [7.0, 8.0, 9.0, 0.0]);
        assert_eq!(gpu.patch_color, [0.1, 0.2, 0.3, 0.4]);
        assert_eq!(gpu.patch_edge_color, [0.5, 0.6, 0.7, 0.8]);
        assert_eq!(gpu.patch_params[0], 2.25);
        assert_eq!(gpu.transmission, [0.1, 0.2, 1.7, 0.0]);
        assert_eq!(gpu.absorption, [0.3, 0.4, 0.5, 0.8]);
    }

    #[test]
    fn timestamp_delta_handles_hardware_counter_wraparound() {
        assert_eq!(timestamp_delta(10, 25, 64), 15);
        assert_eq!(timestamp_delta(250, 5, 8), 11);
    }
}
