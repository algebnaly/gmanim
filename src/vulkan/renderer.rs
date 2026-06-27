use crate::mobjects::mesh_2d::{TriangleMesh2D, Vertex2D};
use crate::mobjects::mesh_3d::{TriangleMesh3D, Vertex};
use crate::video_backend::vulkan_h264::VulkanVideoFrame;
use crate::vulkan::context::VulkanContext;
use ash::vk;
use std::sync::Arc;

// The encoder keeps 8 frames in flight; one extra image prevents the renderer
// from overwriting an image before submit-side backpressure can release a slot.
const VIDEO_NV12_IMAGE_COUNT: usize = 9;
const VK_FORMAT_G8_B8R8_2PLANE_420_UNORM_RAW: i32 = 1_000_156_003;
const VK_IMAGE_USAGE_VIDEO_ENCODE_SRC_BIT_KHR_RAW: u32 = 0x0000_4000;
const VK_IMAGE_ASPECT_PLANE_0_BIT_RAW: u32 = 0x0000_0010;
const VK_IMAGE_ASPECT_PLANE_1_BIT_RAW: u32 = 0x0000_0020;

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
        let image_info = vk::ImageCreateInfo {
            s_type: vk::StructureType::IMAGE_CREATE_INFO,
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

        let color_view =
            Self::create_view(ctx, vk_image, format, vk::ImageAspectFlags::COLOR, usage);
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

pub struct RenderCache {
    pub width: u32,
    pub height: u32,
    pub texture: Image,
    pub depth_texture: Image,
    pub msaa_texture: Image,
    pub msaa_depth_texture: Image,
    pub resolved_texture: Image,
    pub output_buffers: [Buffer; 3],
    pub nv12_output_buffers: [Buffer; 3],
    pub nv12_descriptor_sets: [vk::DescriptorSet; 3],
    pub video_nv12_images: Vec<VideoNv12Image>,
    pub video_nv12_descriptor_sets: Vec<vk::DescriptorSet>,
    pub video_nv12_layouts: Vec<vk::ImageLayout>,
    pub current_frame: usize,
    pub compute_descriptor_set: vk::DescriptorSet,
    pub raster_descriptor_set: vk::DescriptorSet,
    pub raster_descriptor_set_2d: vk::DescriptorSet,
    pub composite_descriptor_set: vk::DescriptorSet,
    pub padded_bytes_per_row: u32,
    pub framebuffer: vk::Framebuffer,
    pub rgba_preview_buffer: Vec<u8>,
}

impl RenderCache {
    pub fn destroy(&mut self, ctx: &VulkanContext) {
        unsafe {
            ctx.device.destroy_framebuffer(self.framebuffer, None);
        }
        self.texture.destroy(ctx);
        self.depth_texture.destroy(ctx);
        self.msaa_texture.destroy(ctx);
        self.msaa_depth_texture.destroy(ctx);
        self.resolved_texture.destroy(ctx);
        for buf in &mut self.output_buffers {
            buf.destroy(ctx);
        }
        for buf in &mut self.nv12_output_buffers {
            buf.destroy(ctx);
        }
        for image in &mut self.video_nv12_images {
            image.destroy(ctx);
        }
    }
}

pub struct FrameData {
    pub command_pool: vk::CommandPool,
    pub command_buffer: vk::CommandBuffer,
    pub fence: vk::Fence,
}

#[derive(Clone, Copy, Debug)]
pub struct RenderOutputs {
    pub cpu_nv12: bool,
    pub vulkan_video: bool,
    pub cpu_rgba: bool,
}

impl RenderOutputs {
    pub const ALL: Self = Self {
        cpu_nv12: true,
        vulkan_video: true,
        cpu_rgba: true,
    };

    pub const VULKAN_VIDEO_ONLY: Self = Self {
        cpu_nv12: false,
        vulkan_video: true,
        cpu_rgba: false,
    };
}

pub struct VulkanRenderer {
    ctx: Arc<VulkanContext>,

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
    nv12_pipeline_layout: vk::PipelineLayout,
    nv12_pipeline: vk::Pipeline,
    video_nv12_pipeline_layout: vk::PipelineLayout,
    video_nv12_pipeline: vk::Pipeline,

    render_pass: vk::RenderPass,
    raster_pipeline_layout: vk::PipelineLayout,
    raster_pipeline: vk::Pipeline,

    vertex_buffer: Buffer,
    index_buffer: Buffer,
    camera_buffer: Buffer,
    buffer_3d: Buffer,
    nv12_constants_buffer: Buffer,

    raster_pipeline_layout_2d: vk::PipelineLayout,
    raster_pipeline_2d: vk::Pipeline,
    vertex_buffer_2d: Buffer,
    index_buffer_2d: Buffer,
    camera_buffer_2d: Buffer,

    frame_data: [FrameData; 3],

    cache: std::sync::Mutex<Option<RenderCache>>,
}

impl VulkanRenderer {
    pub fn new(ctx: Arc<VulkanContext>) -> Self {
        let compute_shader = compile_wgsl_full(&ctx, include_str!("shader.wgsl"));
        let raster_shader = compile_wgsl_full(&ctx, include_str!("raster_shader.wgsl"));
        let raster_shader_2d = compile_wgsl_full(&ctx, include_str!("raster_shader_2d.wgsl"));
        let nv12_shader = compile_wgsl_full(&ctx, include_str!("rgba_to_nv12.wgsl"));
        let video_nv12_shader = compile_wgsl_full(&ctx, include_str!("rgba_to_nv12_image.wgsl"));
        let composite_shader = compile_wgsl_full(&ctx, include_str!("composite_shader.wgsl"));

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
                descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
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
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE
                    | vk::ShaderStageFlags::VERTEX
                    | vk::ShaderStageFlags::FRAGMENT,
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: 2,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
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
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
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

        let color_attachment = vk::AttachmentDescription {
            format: vk::Format::R8G8B8A8_UNORM,
            samples: vk::SampleCountFlags::TYPE_8,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::DONT_CARE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            ..Default::default()
        };

        let depth_attachment = vk::AttachmentDescription {
            format: vk::Format::D32_SFLOAT,
            samples: vk::SampleCountFlags::TYPE_8,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::DONT_CARE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            ..Default::default()
        };

        let resolve_attachment = vk::AttachmentDescription {
            format: vk::Format::R8G8B8A8_UNORM,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::GENERAL,
            ..Default::default()
        };

        let color_attachment_ref = vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        };

        let depth_attachment_ref = vk::AttachmentReference {
            attachment: 1,
            layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };

        let resolve_attachment_ref = vk::AttachmentReference {
            attachment: 2,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        };

        let subpass = vk::SubpassDescription {
            pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
            color_attachment_count: 1,
            p_color_attachments: &color_attachment_ref,
            p_depth_stencil_attachment: &depth_attachment_ref,
            p_resolve_attachments: &resolve_attachment_ref,
            ..Default::default()
        };

        let attachments = [color_attachment, depth_attachment, resolve_attachment];
        let render_pass_info = vk::RenderPassCreateInfo {
            s_type: vk::StructureType::RENDER_PASS_CREATE_INFO,
            attachment_count: attachments.len() as u32,
            p_attachments: attachments.as_ptr(),
            subpass_count: 1,
            p_subpasses: &subpass,
            ..Default::default()
        };

        let render_pass = unsafe {
            ctx.device
                .create_render_pass(&render_pass_info, None)
                .unwrap()
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
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
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
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
            depth_bias_enable: vk::FALSE,
            ..Default::default()
        };

        let multisampling = vk::PipelineMultisampleStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            sample_shading_enable: vk::FALSE,
            rasterization_samples: vk::SampleCountFlags::TYPE_8,
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

        let raster_pipeline_info = vk::GraphicsPipelineCreateInfo {
            s_type: vk::StructureType::GRAPHICS_PIPELINE_CREATE_INFO,
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
            render_pass,
            subpass: 0,
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

        let vertex_binding_description_2d = vk::VertexInputBindingDescription {
            binding: 0,
            stride: std::mem::size_of::<Vertex2D>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        };
        let vertex_attribute_descriptions_2d = [
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: memoffset::offset_of!(Vertex2D, position) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 1,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: memoffset::offset_of!(Vertex2D, color) as u32,
            },
        ];
        let vertex_input_info_2d = vk::PipelineVertexInputStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            vertex_binding_description_count: 1,
            p_vertex_binding_descriptions: &vertex_binding_description_2d,
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
            render_pass,
            subpass: 0,
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

        let descriptor_pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 80,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 40,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 40,
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

        let vertex_buffer = Buffer::new(
            &ctx,
            (std::mem::size_of::<Vertex>() * 1_000_000) as u64,
            vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            gpu_allocator::MemoryLocation::CpuToGpu,
        );
        let index_buffer = Buffer::new(
            &ctx,
            (std::mem::size_of::<u32>() * 3_000_000) as u64,
            vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            gpu_allocator::MemoryLocation::CpuToGpu,
        );
        let camera_buffer = Buffer::new(
            &ctx,
            std::mem::size_of::<CameraUniform>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            gpu_allocator::MemoryLocation::CpuToGpu,
        );
        let buffer_3d = Buffer::new(
            &ctx,
            (std::mem::size_of::<PrimitiveData3D>() * 10000) as u64,
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
            (std::mem::size_of::<Vertex2D>() * 1_000_000) as u64,
            vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            gpu_allocator::MemoryLocation::CpuToGpu,
        );
        let index_buffer_2d = Buffer::new(
            &ctx,
            (std::mem::size_of::<u32>() * 3_000_000) as u64,
            vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            gpu_allocator::MemoryLocation::CpuToGpu,
        );
        let camera_buffer_2d = Buffer::new(
            &ctx,
            std::mem::size_of::<CameraUniform2D>() as u64,
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
            FrameData {
                command_pool,
                command_buffer,
                fence,
            }
        });

        unsafe {
            ctx.device.destroy_shader_module(compute_shader, None);
            ctx.device.destroy_shader_module(raster_shader, None);
            ctx.device.destroy_shader_module(raster_shader_2d, None);
            ctx.device.destroy_shader_module(nv12_shader, None);
            ctx.device.destroy_shader_module(video_nv12_shader, None);
            ctx.device.destroy_shader_module(composite_shader, None);
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
            nv12_pipeline_layout,
            nv12_pipeline,
            video_nv12_pipeline_layout,
            video_nv12_pipeline,
            render_pass,
            raster_pipeline_layout,
            raster_pipeline,
            vertex_buffer,
            index_buffer,
            camera_buffer,
            buffer_3d,
            nv12_constants_buffer,
            raster_pipeline_layout_2d,
            raster_pipeline_2d,
            vertex_buffer_2d,
            index_buffer_2d,
            camera_buffer_2d,
            frame_data,
            cache: std::sync::Mutex::new(None),
        }
    }

    pub fn render_scene(
        &mut self,
        scene: &crate::Scene,
        scene_config: &crate::SceneConfig,
        output: Option<&mut [u8]>,
    ) {
        self.render_scene_with_outputs(scene, scene_config, output, RenderOutputs::ALL);
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
                let (o_left, o_right, o_bottom, o_top) = scene.camera.ortho_params();
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

        let mut mesh_vertices_2d = Vec::new();
        let mut mesh_indices_2d = Vec::new();

        struct VulkanDataCollector<'a> {
            primitives_3d: &'a mut Vec<PrimitiveData3D>,
            mesh_vertices: &'a mut Vec<Vertex>,
            mesh_indices: &'a mut Vec<u32>,
            mesh_vertices_2d: &'a mut Vec<Vertex2D>,
            mesh_indices_2d: &'a mut Vec<u32>,
        }

        impl<'a> crate::mobjects::RenderVisitor for VulkanDataCollector<'a> {
            fn push_mesh_2d(
                &mut self,
                mesh: &crate::mobjects::mesh_2d::TriangleMesh2D,
                transform: nalgebra::Matrix4<crate::GMFloat>,
            ) {
                let base_index = self.mesh_vertices_2d.len() as u32;
                let mesh_mat = transform * mesh.model_matrix;
                for v in &mesh.vertices {
                    let pos = nalgebra::Point3::new(
                        v.position[0] as crate::GMFloat,
                        v.position[1] as crate::GMFloat,
                        0.0,
                    );
                    let t_pos = mesh_mat.transform_point(&pos);
                    self.mesh_vertices_2d.push(Vertex2D {
                        position: [t_pos.x as f32, t_pos.y as f32],
                        color: v.color,
                    });
                }
                for i in &mesh.indices {
                    self.mesh_indices_2d.push(*i + base_index);
                }
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
            mesh_vertices_2d: &mut mesh_vertices_2d,
            mesh_indices_2d: &mut mesh_indices_2d,
        };

        for m in &scene.mobjects {
            m.borrow()
                .submit_to_renderer(&mut collector, nalgebra::Matrix4::identity());
        }

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
                let mut p = crate::camera::Projection::perspective_wgpu(
                    scene.camera.fov() as f32,
                    output_w / output_h,
                    scene.camera.perspective_params().0 as f32,
                    scene.camera.perspective_params().1 as f32,
                );
                // Correct for Vulkan's inverted Y in clip space
                p[5] *= -1.0;
                p
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
            &mesh_vertices_2d,
            &mesh_indices_2d,
            output,
            outputs,
        );
    }

    pub fn render(
        &mut self,
        width: u32,
        height: u32,
        camera_uniform: &CameraUniform,
        camera_uniform_2d: &CameraUniform2D,
        objects_3d: &[PrimitiveData3D],
        mesh_vertices: &[Vertex],
        mesh_indices: &[u32],
        mesh_vertices_2d: &[Vertex2D],
        mesh_indices_2d: &[u32],
        output: Option<&mut [u8]>,
        outputs: RenderOutputs,
    ) {
        self.camera_buffer
            .write_bytes(0, bytemuck::bytes_of(camera_uniform));

        self.camera_buffer_2d
            .write_bytes(0, bytemuck::bytes_of(camera_uniform_2d));
        if !mesh_vertices_2d.is_empty() {
            let bytes_v = bytemuck::cast_slice(mesh_vertices_2d);
            let len = (self.vertex_buffer_2d.size as usize).min(bytes_v.len());
            self.vertex_buffer_2d.write_bytes(0, &bytes_v[..len]);
        }
        if !mesh_indices_2d.is_empty() {
            let bytes_i = bytemuck::cast_slice(mesh_indices_2d);
            let len = (self.index_buffer_2d.size as usize).min(bytes_i.len());
            self.index_buffer_2d.write_bytes(0, &bytes_i[..len]);
        }

        self.nv12_constants_buffer.write_bytes(
            0,
            bytemuck::bytes_of(&Nv12Constants {
                width,
                height,
                _padding: [0; 2],
            }),
        );
        if !objects_3d.is_empty() {
            let bytes_3d = bytemuck::cast_slice(objects_3d);
            let len = (self.buffer_3d.size as usize).min(bytes_3d.len());
            self.buffer_3d.write_bytes(0, &bytes_3d[..len]);
        }
        if !mesh_vertices.is_empty() {
            let bytes_v = bytemuck::cast_slice(mesh_vertices);
            let len = (self.vertex_buffer.size as usize).min(bytes_v.len());
            self.vertex_buffer.write_bytes(0, &bytes_v[..len]);
        }
        if !mesh_indices.is_empty() {
            let bytes_i = bytemuck::cast_slice(mesh_indices);
            let len = (self.index_buffer.size as usize).min(bytes_i.len());
            self.index_buffer.write_bytes(0, &bytes_i[..len]);
        }

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

            let texture = Image::new(
                &self.ctx,
                width,
                height,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::TRANSFER_SRC,
                vk::ImageAspectFlags::COLOR,
                vk::SampleCountFlags::TYPE_1,
            );
            let msaa_texture = Image::new(
                &self.ctx,
                width,
                height,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageUsageFlags::COLOR_ATTACHMENT,
                vk::ImageAspectFlags::COLOR,
                vk::SampleCountFlags::TYPE_8,
            );
            let msaa_depth_texture = Image::new(
                &self.ctx,
                width,
                height,
                vk::Format::D32_SFLOAT,
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                vk::ImageAspectFlags::DEPTH,
                vk::SampleCountFlags::TYPE_8,
            );
            let resolved_texture = Image::new(
                &self.ctx,
                width,
                height,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
                vk::ImageAspectFlags::COLOR,
                vk::SampleCountFlags::TYPE_1,
            );
            let depth_texture = Image::new(
                &self.ctx,
                width,
                height,
                vk::Format::D32_SFLOAT,
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                vk::ImageAspectFlags::DEPTH,
                vk::SampleCountFlags::TYPE_1,
            );

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

            let alloc_info = vk::DescriptorSetAllocateInfo {
                s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
                descriptor_pool: self.descriptor_pool,
                descriptor_set_count: 1,
                p_set_layouts: &self.compute_descriptor_set_layout,
                ..Default::default()
            };
            let compute_descriptor_set = unsafe {
                self.ctx
                    .device
                    .allocate_descriptor_sets(&alloc_info)
                    .unwrap()[0]
            };

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

            let alloc_info_composite = vk::DescriptorSetAllocateInfo {
                s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
                descriptor_pool: self.descriptor_pool,
                descriptor_set_count: 1,
                p_set_layouts: &self.composite_descriptor_set_layout,
                ..Default::default()
            };
            let composite_descriptor_set = unsafe {
                self.ctx
                    .device
                    .allocate_descriptor_sets(&alloc_info_composite)
                    .unwrap()[0]
            };

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

            let video_nv12_images = (0..VIDEO_NV12_IMAGE_COUNT)
                .map(|_| VideoNv12Image::new(&self.ctx, width, height))
                .collect::<Vec<_>>();
            let video_nv12_layouts = vec![vk::ImageLayout::UNDEFINED; VIDEO_NV12_IMAGE_COUNT];
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

            let image_info = vk::DescriptorImageInfo {
                image_view: texture.view,
                image_layout: vk::ImageLayout::GENERAL,
                ..Default::default()
            };
            let image_info_resolved = vk::DescriptorImageInfo {
                image_view: resolved_texture.view,
                image_layout: vk::ImageLayout::GENERAL,
                ..Default::default()
            };
            let camera_buffer_info = vk::DescriptorBufferInfo {
                buffer: self.camera_buffer.vk_buffer,
                offset: 0,
                range: vk::WHOLE_SIZE,
                ..Default::default()
            };
            let buffer_3d_info = vk::DescriptorBufferInfo {
                buffer: self.buffer_3d.vk_buffer,
                offset: 0,
                range: vk::WHOLE_SIZE,
                ..Default::default()
            };

            let camera_buffer_2d_info = vk::DescriptorBufferInfo {
                buffer: self.camera_buffer_2d.vk_buffer,
                offset: 0,
                range: vk::WHOLE_SIZE,
                ..Default::default()
            };

            let mut write_descriptor_sets = vec![
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: compute_descriptor_set,
                    dst_binding: 0,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &image_info,
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: compute_descriptor_set,
                    dst_binding: 1,
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: 1,
                    p_buffer_info: &camera_buffer_info,
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: compute_descriptor_set,
                    dst_binding: 2,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: 1,
                    p_buffer_info: &buffer_3d_info,
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: raster_descriptor_set,
                    dst_binding: 1,
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: 1,
                    p_buffer_info: &camera_buffer_info,
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: raster_descriptor_set_2d,
                    dst_binding: 0,
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: 1,
                    p_buffer_info: &camera_buffer_2d_info,
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: composite_descriptor_set,
                    dst_binding: 0,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &image_info,
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: composite_descriptor_set,
                    dst_binding: 1,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &image_info_resolved,
                    ..Default::default()
                },
            ];

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
                    p_image_info: &image_info,
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

            let mut video_nv12_input_infos = Vec::new();
            let mut video_nv12_y_infos = Vec::new();
            let mut video_nv12_uv_infos = Vec::new();
            for image in &video_nv12_images {
                video_nv12_input_infos.push(vk::DescriptorImageInfo {
                    image_view: texture.view,
                    image_layout: vk::ImageLayout::GENERAL,
                    ..Default::default()
                });
                video_nv12_y_infos.push(vk::DescriptorImageInfo {
                    image_view: image.y_view,
                    image_layout: vk::ImageLayout::GENERAL,
                    ..Default::default()
                });
                video_nv12_uv_infos.push(vk::DescriptorImageInfo {
                    image_view: image.uv_view,
                    image_layout: vk::ImageLayout::GENERAL,
                    ..Default::default()
                });
            }
            for i in 0..VIDEO_NV12_IMAGE_COUNT {
                write_descriptor_sets.push(vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: video_nv12_descriptor_sets[i],
                    dst_binding: 0,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &video_nv12_input_infos[i],
                    ..Default::default()
                });
                write_descriptor_sets.push(vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: video_nv12_descriptor_sets[i],
                    dst_binding: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &video_nv12_y_infos[i],
                    ..Default::default()
                });
                write_descriptor_sets.push(vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: video_nv12_descriptor_sets[i],
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

            let attachments = [
                msaa_texture.view,
                msaa_depth_texture.view,
                resolved_texture.view,
            ];
            let framebuffer_info = vk::FramebufferCreateInfo {
                s_type: vk::StructureType::FRAMEBUFFER_CREATE_INFO,
                render_pass: self.render_pass,
                attachment_count: attachments.len() as u32,
                p_attachments: attachments.as_ptr(),
                width,
                height,
                layers: 1,
                ..Default::default()
            };
            let framebuffer = unsafe {
                self.ctx
                    .device
                    .create_framebuffer(&framebuffer_info, None)
                    .unwrap()
            };

            *cache_guard = Some(RenderCache {
                width,
                height,
                texture,
                depth_texture,
                msaa_texture,
                msaa_depth_texture,
                resolved_texture,
                output_buffers,
                nv12_output_buffers,
                nv12_descriptor_sets,
                video_nv12_images,
                video_nv12_descriptor_sets,
                video_nv12_layouts,
                current_frame: 0,
                compute_descriptor_set,
                raster_descriptor_set,
                raster_descriptor_set_2d,
                composite_descriptor_set,
                padded_bytes_per_row,
                framebuffer,
                rgba_preview_buffer: vec![0; (width * height * 4) as usize],
            });
        }

        let mut cache = cache_guard.as_mut().unwrap();
        let frame_idx = cache.current_frame % 3;
        let video_frame_idx = cache.current_frame % cache.video_nv12_images.len();
        let fd = &self.frame_data[frame_idx];

        unsafe {
            self.ctx
                .device
                .wait_for_fences(std::slice::from_ref(&fd.fence), true, std::u64::MAX)
                .unwrap();
            self.ctx
                .device
                .reset_fences(std::slice::from_ref(&fd.fence))
                .unwrap();
            self.ctx
                .device
                .reset_command_pool(fd.command_pool, vk::CommandPoolResetFlags::empty())
                .unwrap();

            let begin_info = vk::CommandBufferBeginInfo {
                s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
                flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                ..Default::default()
            };
            self.ctx
                .device
                .begin_command_buffer(fd.command_buffer, &begin_info)
                .unwrap();

            let barrier = vk::ImageMemoryBarrier {
                s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: vk::ImageLayout::GENERAL,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image: cache.texture.vk_image,
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
                std::slice::from_ref(&barrier),
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
                std::slice::from_ref(&cache.compute_descriptor_set),
                &[],
            );
            let workgroup_x = (width + 15) / 16;
            let workgroup_y = (height + 15) / 16;
            self.ctx
                .device
                .cmd_dispatch(fd.command_buffer, workgroup_x, workgroup_y, 1);

            let clear_values = [
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 0.0],
                    },
                },
                vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                },
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 0.0],
                    },
                },
            ];

            let render_pass_begin_info = vk::RenderPassBeginInfo {
                s_type: vk::StructureType::RENDER_PASS_BEGIN_INFO,
                render_pass: self.render_pass,
                framebuffer: cache.framebuffer,
                render_area: vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D { width, height },
                },
                clear_value_count: clear_values.len() as u32,
                p_clear_values: clear_values.as_ptr(),
                ..Default::default()
            };

            self.ctx.device.cmd_begin_render_pass(
                fd.command_buffer,
                &render_pass_begin_info,
                vk::SubpassContents::INLINE,
            );

            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: width as f32,
                height: height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            self.ctx
                .device
                .cmd_set_viewport(fd.command_buffer, 0, std::slice::from_ref(&viewport));
            let scissor = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D { width, height },
            };
            self.ctx
                .device
                .cmd_set_scissor(fd.command_buffer, 0, std::slice::from_ref(&scissor));

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
                    &[],
                );
                self.ctx.device.cmd_bind_vertex_buffers(
                    fd.command_buffer,
                    0,
                    std::slice::from_ref(&self.vertex_buffer.vk_buffer),
                    &[0],
                );
                self.ctx.device.cmd_bind_index_buffer(
                    fd.command_buffer,
                    self.index_buffer.vk_buffer,
                    0,
                    vk::IndexType::UINT32,
                );
                let indices_to_draw =
                    (mesh_indices.len() as u32).min((self.index_buffer.size / 4) as u32);
                self.ctx
                    .device
                    .cmd_draw_indexed(fd.command_buffer, indices_to_draw, 1, 0, 0, 0);
            }

            if !mesh_indices_2d.is_empty() {
                self.ctx.device.cmd_bind_pipeline(
                    fd.command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.raster_pipeline_2d,
                );
                self.ctx.device.cmd_bind_descriptor_sets(
                    fd.command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.raster_pipeline_layout_2d,
                    0,
                    std::slice::from_ref(&cache.raster_descriptor_set_2d),
                    &[],
                );
                self.ctx.device.cmd_bind_vertex_buffers(
                    fd.command_buffer,
                    0,
                    std::slice::from_ref(&self.vertex_buffer_2d.vk_buffer),
                    &[0],
                );
                self.ctx.device.cmd_bind_index_buffer(
                    fd.command_buffer,
                    self.index_buffer_2d.vk_buffer,
                    0,
                    vk::IndexType::UINT32,
                );
                let indices_to_draw =
                    (mesh_indices_2d.len() as u32).min((self.index_buffer_2d.size / 4) as u32);
                self.ctx
                    .device
                    .cmd_draw_indexed(fd.command_buffer, indices_to_draw, 1, 0, 0, 0);
            }

            self.ctx.device.cmd_end_render_pass(fd.command_buffer);

            let composite_barrier_1 = vk::ImageMemoryBarrier {
                s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
                old_layout: vk::ImageLayout::GENERAL,
                new_layout: vk::ImageLayout::GENERAL,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image: cache.resolved_texture.vk_image,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                dst_access_mask: vk::AccessFlags::SHADER_READ,
                ..Default::default()
            };
            let composite_barrier_2 = vk::ImageMemoryBarrier {
                s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
                old_layout: vk::ImageLayout::GENERAL,
                new_layout: vk::ImageLayout::GENERAL,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image: cache.texture.vk_image,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                src_access_mask: vk::AccessFlags::SHADER_WRITE,
                dst_access_mask: vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                ..Default::default()
            };

            self.ctx.device.cmd_pipeline_barrier(
                fd.command_buffer,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[composite_barrier_1, composite_barrier_2],
            );

            self.ctx.device.cmd_bind_pipeline(
                fd.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.composite_pipeline,
            );
            self.ctx.device.cmd_bind_descriptor_sets(
                fd.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.composite_pipeline_layout,
                0,
                std::slice::from_ref(&cache.composite_descriptor_set),
                &[],
            );
            let workgroup_x = (width + 15) / 16;
            let workgroup_y = (height + 15) / 16;
            self.ctx
                .device
                .cmd_dispatch(fd.command_buffer, workgroup_x, workgroup_y, 1);

            let postprocess_barrier = vk::ImageMemoryBarrier {
                s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
                old_layout: vk::ImageLayout::GENERAL,
                new_layout: vk::ImageLayout::GENERAL,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image: cache.texture.vk_image,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                src_access_mask: vk::AccessFlags::SHADER_WRITE,
                dst_access_mask: vk::AccessFlags::SHADER_READ,
                ..Default::default()
            };

            self.ctx.device.cmd_pipeline_barrier(
                fd.command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                std::slice::from_ref(&postprocess_barrier),
            );

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

            if outputs.vulkan_video {
                let video_nv12_image = &cache.video_nv12_images[video_frame_idx];
                let video_nv12_barrier = vk::ImageMemoryBarrier {
                    s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
                    old_layout: cache.video_nv12_layouts[video_frame_idx],
                    new_layout: vk::ImageLayout::GENERAL,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    image: video_nv12_image.vk_image,
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
                    self.video_nv12_pipeline,
                );
                self.ctx.device.cmd_bind_descriptor_sets(
                    fd.command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    self.video_nv12_pipeline_layout,
                    0,
                    std::slice::from_ref(&cache.video_nv12_descriptor_sets[video_frame_idx]),
                    &[],
                );
                let workgroup_x = (width + 15) / 16;
                let workgroup_y = (height + 15) / 16;
                self.ctx
                    .device
                    .cmd_dispatch(fd.command_buffer, workgroup_x, workgroup_y, 1);
                cache.video_nv12_layouts[video_frame_idx] = vk::ImageLayout::GENERAL;
            }

            if outputs.cpu_rgba {
                let image_barrier2 = vk::ImageMemoryBarrier {
                    s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
                    old_layout: vk::ImageLayout::GENERAL,
                    new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    image: cache.texture.vk_image,
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    src_access_mask: vk::AccessFlags::SHADER_READ,
                    dst_access_mask: vk::AccessFlags::TRANSFER_READ,
                    ..Default::default()
                };

                self.ctx.device.cmd_pipeline_barrier(
                    fd.command_buffer,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    std::slice::from_ref(&image_barrier2),
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
                    cache.texture.vk_image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    cache.output_buffers[frame_idx].vk_buffer,
                    std::slice::from_ref(&copy_region),
                );
            }

            self.ctx
                .device
                .end_command_buffer(fd.command_buffer)
                .unwrap();

            let submit_info = vk::SubmitInfo {
                s_type: vk::StructureType::SUBMIT_INFO,
                command_buffer_count: 1,
                p_command_buffers: &fd.command_buffer,
                ..Default::default()
            };
            self.ctx
                .device
                .queue_submit(self.ctx.queue, std::slice::from_ref(&submit_info), fd.fence)
                .unwrap();
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

    pub fn get_vulkan_video_frame(&self) -> Option<VulkanVideoFrame> {
        let guard = self.cache.lock().unwrap();
        if let Some(cache) = guard.as_ref() {
            if cache.current_frame == 0 || cache.video_nv12_images.is_empty() {
                return None;
            }
            let read_frame_idx = (cache.current_frame - 1) % 3;
            let video_frame_idx = (cache.current_frame - 1) % cache.video_nv12_images.len();
            let read_fd = &self.frame_data[read_frame_idx];

            unsafe {
                self.ctx
                    .device
                    .wait_for_fences(std::slice::from_ref(&read_fd.fence), true, std::u64::MAX)
                    .unwrap();
            }

            let image = &cache.video_nv12_images[video_frame_idx];
            return Some(VulkanVideoFrame {
                image: image.vk_image,
                image_view: image.color_view,
                image_layout: cache.video_nv12_layouts[video_frame_idx],
                format: image.format,
                width: image.width,
                height: image.height,
            });
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
