use ash::vk;

use std::sync::Arc;

use crate::vulkan::context::VulkanContext;

use super::frame::{RENDER_FRAME_COUNT, TrackedImageState};
use super::prepared_frame::FrameRequirements;
use super::record::RecordingPlan;
use super::video_output::{VIDEO_NV12_IMAGE_COUNT, VideoNv12Slot};
use super::{Buffer, DescriptorPool, Image, Nv12Constants, PipelineSet, msaa_to_vk_sample_count};

pub(super) struct TargetCacheResources<'a> {
    pub(super) ctx: &'a Arc<VulkanContext>,
    pub(super) descriptor_pool: &'a Arc<DescriptorPool>,
    pub(super) pipelines: &'a PipelineSet,
    pub(super) msaa_samples: u32,
    pub(super) ssaa_factor: u32,
    pub(super) environment_map: &'a Image,
    pub(super) environment_sampler: vk::Sampler,
    pub(super) camera_buffer: &'a Buffer,
    pub(super) material_buffer_3d: &'a Buffer,
    pub(super) primitive_buffer: &'a Buffer,
    pub(super) grid_buffer_3d: &'a Buffer,
    pub(super) camera_buffer_2d: &'a Buffer,
    pub(super) nv12_constants_buffer: &'a Buffer,
    pub(super) tone_map_factor_buffer: &'a Buffer,
    pub(super) camera_buffer_stride: u64,
    pub(super) material_buffer_3d_stride: u64,
    pub(super) primitive_buffer_stride: u64,
    pub(super) grid_buffer_3d_stride: u64,
    pub(super) camera_buffer_2d_stride: u64,
    pub(super) tone_map_factor_stride: u64,
}

pub(super) struct RenderTargetSet {
    pub(super) texture: Image,
    pub(super) texture_state: TrackedImageState,
    pub(super) sdf_normal_coverage: Image,
    pub(super) sdf_normal_coverage_state: TrackedImageState,
    pub(super) sdf_material_id: Image,
    pub(super) sdf_material_id_state: TrackedImageState,
    pub(super) sdf_depth: Image,
    pub(super) sdf_depth_state: TrackedImageState,
    pub(super) resolved_primary_normal_depth: Image,
    pub(super) resolved_primary_normal_depth_state: TrackedImageState,
    pub(super) resolved_primary_albedo_coverage: Image,
    pub(super) resolved_primary_albedo_coverage_state: TrackedImageState,
    pub(super) resolved_secondary_normal_depth: Image,
    pub(super) resolved_secondary_normal_depth_state: TrackedImageState,
    pub(super) resolved_secondary_albedo_coverage: Image,
    pub(super) resolved_secondary_albedo_coverage_state: TrackedImageState,
    pub(super) resolved_material_ids: Image,
    pub(super) resolved_material_ids_state: TrackedImageState,
    pub(super) surface_hdr: Image,
    pub(super) surface_hdr_state: TrackedImageState,
    pub(super) overlay_hdr: Image,
    pub(super) overlay_hdr_state: TrackedImageState,
    pub(super) resolved_texture: Image,
    pub(super) resolved_texture_state: TrackedImageState,
    pub(super) scene_color: Image,
    pub(super) scene_color_state: TrackedImageState,
    pub(super) transparent_back_depth: Image,
    pub(super) transparent_back_depth_state: TrackedImageState,
    pub(super) bloom_ping: Image,
    pub(super) bloom_ping_state: TrackedImageState,
    pub(super) bloom_pong: Image,
    pub(super) bloom_pong_state: TrackedImageState,
    pub(super) bloom_contains_data: bool,
    pub(super) compute_descriptor_set: vk::DescriptorSet,
    pub(super) surface_resolve_descriptor_set: vk::DescriptorSet,
    pub(super) surface_lighting_descriptor_set: vk::DescriptorSet,
    pub(super) surface_composite_descriptor_set: vk::DescriptorSet,
    pub(super) raster_descriptor_set: vk::DescriptorSet,
    pub(super) grid_descriptor_set: vk::DescriptorSet,
    pub(super) composite_descriptor_set: vk::DescriptorSet,
    pub(super) bloom_descriptor_sets: [vk::DescriptorSet; 3],
}

pub(super) struct TargetCache {
    descriptor_pool: Arc<DescriptorPool>,
    pub width: u32,
    pub height: u32,
    pub(super) has_raster_gbuffer: bool,
    pub(super) has_overlay_hdr: bool,
    pub(super) render_targets: [RenderTargetSet; RENDER_FRAME_COUNT],
    pub raster_normal_depth: Image,
    pub(super) raster_normal_depth_state: TrackedImageState,
    pub raster_albedo: Image,
    pub(super) raster_albedo_state: TrackedImageState,
    pub raster_material_id: Image,
    pub(super) raster_material_id_state: TrackedImageState,
    pub msaa_texture: Option<Image>,
    pub(super) msaa_texture_state: TrackedImageState,
    pub msaa_depth_texture: Option<Image>,
    pub(super) msaa_depth_texture_state: TrackedImageState,
    pub output_buffers: [Buffer; RENDER_FRAME_COUNT],
    pub nv12_output_buffers: [Buffer; RENDER_FRAME_COUNT],
    pub nv12_descriptor_sets: [vk::DescriptorSet; RENDER_FRAME_COUNT],
    pub yuv444p_output_buffers: [Buffer; RENDER_FRAME_COUNT],
    pub yuv444p_descriptor_sets: [vk::DescriptorSet; RENDER_FRAME_COUNT],
    pub video_nv12_slots: Vec<VideoNv12Slot>,
    pub current_frame: usize,
    pub raster_descriptor_set_2d: vk::DescriptorSet,
    pub padded_bytes_per_row: u32,
}

impl TargetCache {
    pub(super) fn satisfies(&self, requirements: FrameRequirements) -> bool {
        self.width == requirements.width
            && self.height == requirements.height
            && (!requirements.raster_gbuffer || self.has_raster_gbuffer)
            && (!requirements.overlay_hdr || self.has_overlay_hdr)
    }

    pub(super) fn new(
        requirements: FrameRequirements,
        resources: &TargetCacheResources<'_>,
    ) -> Self {
        let width = requirements.width;
        let height = requirements.height;
        let padded_bytes_per_row = requirements.padded_rgba_row_bytes;
        let needs_raster_gbuffer = requirements.raster_gbuffer;
        let needs_overlay_hdr = requirements.overlay_hdr;
        resources.nv12_constants_buffer.write_bytes(
            0,
            bytemuck::bytes_of(&Nv12Constants {
                width,
                height,
                _padding: [0; 2],
            }),
        );

        let raster_sample_count = msaa_to_vk_sample_count(resources.msaa_samples);
        let raster_gbuffer_width = if needs_raster_gbuffer {
            width * resources.ssaa_factor
        } else {
            1
        };
        let raster_gbuffer_height = if needs_raster_gbuffer {
            height * resources.ssaa_factor
        } else {
            1
        };

        let raster_normal_depth = Image::new(
            resources.ctx,
            raster_gbuffer_width,
            raster_gbuffer_height,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            vk::ImageAspectFlags::COLOR,
            raster_sample_count,
        );
        let raster_albedo = Image::new(
            resources.ctx,
            raster_gbuffer_width,
            raster_gbuffer_height,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            vk::ImageAspectFlags::COLOR,
            raster_sample_count,
        );
        let raster_material_id = Image::new(
            resources.ctx,
            raster_gbuffer_width,
            raster_gbuffer_height,
            vk::Format::R16_UINT,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            vk::ImageAspectFlags::COLOR,
            raster_sample_count,
        );
        let mut render_targets = std::array::from_fn(|_| {
            let texture = Image::new(
                resources.ctx,
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
            let sdf_normal_coverage = Image::new(
                resources.ctx,
                width,
                height,
                vk::Format::R16G16B16A16_SFLOAT,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
                vk::ImageAspectFlags::COLOR,
                vk::SampleCountFlags::TYPE_1,
            );
            let sdf_material_id = Image::new(
                resources.ctx,
                width,
                height,
                vk::Format::R32_UINT,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
                vk::ImageAspectFlags::COLOR,
                vk::SampleCountFlags::TYPE_1,
            );
            let sdf_depth = Image::new(
                resources.ctx,
                width,
                height,
                vk::Format::R32_SFLOAT,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
                vk::ImageAspectFlags::COLOR,
                vk::SampleCountFlags::TYPE_1,
            );
            let resolved_surface_image = |format| {
                Image::new(
                    resources.ctx,
                    width * resources.ssaa_factor,
                    height * resources.ssaa_factor,
                    format,
                    vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
                    vk::ImageAspectFlags::COLOR,
                    vk::SampleCountFlags::TYPE_1,
                )
            };
            let resolved_primary_normal_depth =
                resolved_surface_image(vk::Format::R16G16B16A16_SFLOAT);
            let resolved_primary_albedo_coverage =
                resolved_surface_image(vk::Format::R16G16B16A16_SFLOAT);
            let resolved_secondary_normal_depth =
                resolved_surface_image(vk::Format::R16G16B16A16_SFLOAT);
            let resolved_secondary_albedo_coverage =
                resolved_surface_image(vk::Format::R16G16B16A16_SFLOAT);
            let resolved_material_ids = resolved_surface_image(vk::Format::R32_UINT);
            let surface_hdr = Image::new(
                resources.ctx,
                width * resources.ssaa_factor,
                height * resources.ssaa_factor,
                vk::Format::R16G16B16A16_SFLOAT,
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::STORAGE,
                vk::ImageAspectFlags::COLOR,
                vk::SampleCountFlags::TYPE_1,
            );
            let overlay_hdr = Image::new(
                resources.ctx,
                if needs_overlay_hdr {
                    width * resources.ssaa_factor
                } else {
                    1
                },
                if needs_overlay_hdr {
                    height * resources.ssaa_factor
                } else {
                    1
                },
                vk::Format::R16G16B16A16_SFLOAT,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
                vk::ImageAspectFlags::COLOR,
                vk::SampleCountFlags::TYPE_1,
            );
            let resolved_texture = Image::new(
                resources.ctx,
                width * resources.ssaa_factor,
                height * resources.ssaa_factor,
                vk::Format::R16G16B16A16_SFLOAT,
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::STORAGE,
                vk::ImageAspectFlags::COLOR,
                vk::SampleCountFlags::TYPE_1,
            );
            let scene_color = Image::new(
                resources.ctx,
                width * resources.ssaa_factor,
                height * resources.ssaa_factor,
                vk::Format::R16G16B16A16_SFLOAT,
                vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
                vk::ImageAspectFlags::COLOR,
                vk::SampleCountFlags::TYPE_1,
            );
            let transparent_back_depth = Image::new(
                resources.ctx,
                width * resources.ssaa_factor,
                height * resources.ssaa_factor,
                vk::Format::R32_SFLOAT,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
                vk::ImageAspectFlags::COLOR,
                vk::SampleCountFlags::TYPE_1,
            );
            let bloom_width = (width / 2).max(1);
            let bloom_height = (height / 2).max(1);
            let bloom_usage = vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_DST;
            let bloom_ping = Image::new(
                resources.ctx,
                bloom_width,
                bloom_height,
                vk::Format::R16G16B16A16_SFLOAT,
                bloom_usage,
                vk::ImageAspectFlags::COLOR,
                vk::SampleCountFlags::TYPE_1,
            );
            let bloom_pong = Image::new(
                resources.ctx,
                bloom_width,
                bloom_height,
                vk::Format::R16G16B16A16_SFLOAT,
                bloom_usage,
                vk::ImageAspectFlags::COLOR,
                vk::SampleCountFlags::TYPE_1,
            );
            RenderTargetSet {
                texture,
                texture_state: TrackedImageState::UNDEFINED,
                sdf_normal_coverage,
                sdf_normal_coverage_state: TrackedImageState::UNDEFINED,
                sdf_material_id,
                sdf_material_id_state: TrackedImageState::UNDEFINED,
                sdf_depth,
                sdf_depth_state: TrackedImageState::UNDEFINED,
                resolved_primary_normal_depth,
                resolved_primary_normal_depth_state: TrackedImageState::UNDEFINED,
                resolved_primary_albedo_coverage,
                resolved_primary_albedo_coverage_state: TrackedImageState::UNDEFINED,
                resolved_secondary_normal_depth,
                resolved_secondary_normal_depth_state: TrackedImageState::UNDEFINED,
                resolved_secondary_albedo_coverage,
                resolved_secondary_albedo_coverage_state: TrackedImageState::UNDEFINED,
                resolved_material_ids,
                resolved_material_ids_state: TrackedImageState::UNDEFINED,
                surface_hdr,
                surface_hdr_state: TrackedImageState::UNDEFINED,
                overlay_hdr,
                overlay_hdr_state: TrackedImageState::UNDEFINED,
                resolved_texture,
                resolved_texture_state: TrackedImageState::UNDEFINED,
                scene_color,
                scene_color_state: TrackedImageState::UNDEFINED,
                transparent_back_depth,
                transparent_back_depth_state: TrackedImageState::UNDEFINED,
                bloom_ping,
                bloom_ping_state: TrackedImageState::UNDEFINED,
                bloom_pong,
                bloom_pong_state: TrackedImageState::UNDEFINED,
                bloom_contains_data: false,
                compute_descriptor_set: vk::DescriptorSet::null(),
                surface_resolve_descriptor_set: vk::DescriptorSet::null(),
                surface_lighting_descriptor_set: vk::DescriptorSet::null(),
                surface_composite_descriptor_set: vk::DescriptorSet::null(),
                raster_descriptor_set: vk::DescriptorSet::null(),
                grid_descriptor_set: vk::DescriptorSet::null(),
                composite_descriptor_set: vk::DescriptorSet::null(),
                bloom_descriptor_sets: [vk::DescriptorSet::null(); 3],
            }
        });
        // The multisampled intermediate is allocated lazily on the first
        // frame that actually rasterizes with more than one sample.
        // Analytic-AA 2D frames never touch it, so pure 2D scenes do not
        // pay for a high-resolution 8x MSAA image.
        let msaa_texture = None;
        let output_buffer_size = (padded_bytes_per_row * height) as u64;
        let output_buffers = [
            Buffer::new(
                resources.ctx,
                output_buffer_size,
                vk::BufferUsageFlags::TRANSFER_DST,
                gpu_allocator::MemoryLocation::GpuToCpu,
            ),
            Buffer::new(
                resources.ctx,
                output_buffer_size,
                vk::BufferUsageFlags::TRANSFER_DST,
                gpu_allocator::MemoryLocation::GpuToCpu,
            ),
            Buffer::new(
                resources.ctx,
                output_buffer_size,
                vk::BufferUsageFlags::TRANSFER_DST,
                gpu_allocator::MemoryLocation::GpuToCpu,
            ),
        ];

        let nv12_buffer_size = (width * height * 3 / 2) as u64;
        let nv12_output_buffers = [
            Buffer::new(
                resources.ctx,
                nv12_buffer_size,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                gpu_allocator::MemoryLocation::GpuToCpu,
            ),
            Buffer::new(
                resources.ctx,
                nv12_buffer_size,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                gpu_allocator::MemoryLocation::GpuToCpu,
            ),
            Buffer::new(
                resources.ctx,
                nv12_buffer_size,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                gpu_allocator::MemoryLocation::GpuToCpu,
            ),
        ];

        let yuv444p_buffer_size = (width * height * 3) as u64;
        let yuv444p_output_buffers = [
            Buffer::new(
                resources.ctx,
                yuv444p_buffer_size,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                gpu_allocator::MemoryLocation::GpuToCpu,
            ),
            Buffer::new(
                resources.ctx,
                yuv444p_buffer_size,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                gpu_allocator::MemoryLocation::GpuToCpu,
            ),
            Buffer::new(
                resources.ctx,
                yuv444p_buffer_size,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                gpu_allocator::MemoryLocation::GpuToCpu,
            ),
        ];

        let compute_layouts =
            [resources.pipelines.compute_descriptor_set_layout; RENDER_FRAME_COUNT];
        let alloc_info = vk::DescriptorSetAllocateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptor_pool: resources.descriptor_pool.handle(),
            descriptor_set_count: RENDER_FRAME_COUNT as u32,
            p_set_layouts: compute_layouts.as_ptr(),
            ..Default::default()
        };
        let compute_descriptor_sets = unsafe {
            resources
                .ctx
                .device
                .allocate_descriptor_sets(&alloc_info)
                .unwrap()
        };
        for (targets, descriptor_set) in render_targets.iter_mut().zip(compute_descriptor_sets) {
            targets.compute_descriptor_set = descriptor_set;
        }
        let surface_resolve_layouts =
            [resources.pipelines.surface_resolve_descriptor_set_layout; RENDER_FRAME_COUNT];
        let surface_resolve_descriptor_sets = unsafe {
            resources
                .ctx
                .device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(resources.descriptor_pool.handle())
                        .set_layouts(&surface_resolve_layouts),
                )
                .unwrap()
        };
        for (targets, descriptor_set) in render_targets
            .iter_mut()
            .zip(surface_resolve_descriptor_sets)
        {
            targets.surface_resolve_descriptor_set = descriptor_set;
        }
        let surface_lighting_layouts =
            [resources.pipelines.surface_lighting_descriptor_set_layout; RENDER_FRAME_COUNT];
        let surface_lighting_descriptor_sets = unsafe {
            resources
                .ctx
                .device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(resources.descriptor_pool.handle())
                        .set_layouts(&surface_lighting_layouts),
                )
                .unwrap()
        };
        for (targets, descriptor_set) in render_targets
            .iter_mut()
            .zip(surface_lighting_descriptor_sets)
        {
            targets.surface_lighting_descriptor_set = descriptor_set;
        }
        let surface_composite_layouts =
            [resources.pipelines.surface_composite_descriptor_set_layout; RENDER_FRAME_COUNT];
        let surface_composite_descriptor_sets = unsafe {
            resources
                .ctx
                .device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(resources.descriptor_pool.handle())
                        .set_layouts(&surface_composite_layouts),
                )
                .unwrap()
        };
        for (targets, descriptor_set) in render_targets
            .iter_mut()
            .zip(surface_composite_descriptor_sets)
        {
            targets.surface_composite_descriptor_set = descriptor_set;
        }

        let raster_layouts = [resources.pipelines.raster_descriptor_set_layout; RENDER_FRAME_COUNT];
        let alloc_info_raster = vk::DescriptorSetAllocateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptor_pool: resources.descriptor_pool.handle(),
            descriptor_set_count: RENDER_FRAME_COUNT as u32,
            p_set_layouts: raster_layouts.as_ptr(),
            ..Default::default()
        };
        let raster_descriptor_sets = unsafe {
            resources
                .ctx
                .device
                .allocate_descriptor_sets(&alloc_info_raster)
                .unwrap()
        };
        for (targets, descriptor_set) in render_targets.iter_mut().zip(raster_descriptor_sets) {
            targets.raster_descriptor_set = descriptor_set;
        }

        let grid_layouts = [resources.pipelines.grid_descriptor_set_layout; RENDER_FRAME_COUNT];
        let grid_descriptor_sets = unsafe {
            resources
                .ctx
                .device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(resources.descriptor_pool.handle())
                        .set_layouts(&grid_layouts),
                )
                .unwrap()
        };
        for (targets, descriptor_set) in render_targets.iter_mut().zip(grid_descriptor_sets) {
            targets.grid_descriptor_set = descriptor_set;
        }

        let alloc_info_raster_2d = vk::DescriptorSetAllocateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptor_pool: resources.descriptor_pool.handle(),
            descriptor_set_count: 1,
            p_set_layouts: &resources.pipelines.raster_descriptor_set_layout_2d,
            ..Default::default()
        };
        let raster_descriptor_set_2d = unsafe {
            resources
                .ctx
                .device
                .allocate_descriptor_sets(&alloc_info_raster_2d)
                .unwrap()[0]
        };

        let composite_layouts =
            [resources.pipelines.composite_descriptor_set_layout; RENDER_FRAME_COUNT];
        let alloc_info_composite = vk::DescriptorSetAllocateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptor_pool: resources.descriptor_pool.handle(),
            descriptor_set_count: RENDER_FRAME_COUNT as u32,
            p_set_layouts: composite_layouts.as_ptr(),
            ..Default::default()
        };
        let composite_descriptor_sets = unsafe {
            resources
                .ctx
                .device
                .allocate_descriptor_sets(&alloc_info_composite)
                .unwrap()
        };
        for (targets, descriptor_set) in render_targets.iter_mut().zip(composite_descriptor_sets) {
            targets.composite_descriptor_set = descriptor_set;
        }
        let bloom_layouts =
            [resources.pipelines.bloom_descriptor_set_layout; RENDER_FRAME_COUNT * 3];
        let bloom_alloc_info = vk::DescriptorSetAllocateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptor_pool: resources.descriptor_pool.handle(),
            descriptor_set_count: bloom_layouts.len() as u32,
            p_set_layouts: bloom_layouts.as_ptr(),
            ..Default::default()
        };
        let bloom_descriptor_sets = unsafe {
            resources
                .ctx
                .device
                .allocate_descriptor_sets(&bloom_alloc_info)
                .unwrap()
        };
        let (bloom_descriptor_sets, remainder) = bloom_descriptor_sets.as_chunks::<3>();
        debug_assert!(remainder.is_empty());
        for (targets, sets) in render_targets.iter_mut().zip(bloom_descriptor_sets) {
            targets.bloom_descriptor_sets = *sets;
        }

        let nv12_layouts = [resources.pipelines.nv12_descriptor_set_layout; 3];
        let nv12_alloc_info = vk::DescriptorSetAllocateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptor_pool: resources.descriptor_pool.handle(),
            descriptor_set_count: 3,
            p_set_layouts: nv12_layouts.as_ptr(),
            ..Default::default()
        };
        let nv12_descriptor_sets_vec = unsafe {
            resources
                .ctx
                .device
                .allocate_descriptor_sets(&nv12_alloc_info)
                .unwrap()
        };
        let nv12_descriptor_sets: [vk::DescriptorSet; 3] =
            nv12_descriptor_sets_vec.try_into().unwrap();

        let yuv444p_alloc_info = vk::DescriptorSetAllocateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptor_pool: resources.descriptor_pool.handle(),
            descriptor_set_count: 3,
            p_set_layouts: nv12_layouts.as_ptr(),
            ..Default::default()
        };
        let yuv444p_descriptor_sets_vec = unsafe {
            resources
                .ctx
                .device
                .allocate_descriptor_sets(&yuv444p_alloc_info)
                .unwrap()
        };
        let yuv444p_descriptor_sets: [vk::DescriptorSet; 3] =
            yuv444p_descriptor_sets_vec.try_into().unwrap();

        let mut video_nv12_slots = (0..VIDEO_NV12_IMAGE_COUNT)
            .map(|_| VideoNv12Slot::new(resources.ctx, width, height))
            .collect::<Vec<_>>();
        let video_nv12_set_layouts =
            [resources.pipelines.video_nv12_descriptor_set_layout; VIDEO_NV12_IMAGE_COUNT];
        let video_nv12_alloc_info = vk::DescriptorSetAllocateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptor_pool: resources.descriptor_pool.handle(),
            descriptor_set_count: VIDEO_NV12_IMAGE_COUNT as u32,
            p_set_layouts: video_nv12_set_layouts.as_ptr(),
            ..Default::default()
        };
        let video_nv12_descriptor_sets = unsafe {
            resources
                .ctx
                .device
                .allocate_descriptor_sets(&video_nv12_alloc_info)
                .unwrap()
        };
        for (slot, descriptor_set) in video_nv12_slots.iter_mut().zip(video_nv12_descriptor_sets) {
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
        let sdf_normal_coverage_infos: Vec<_> = render_targets
            .iter()
            .map(|targets| vk::DescriptorImageInfo {
                image_view: targets.sdf_normal_coverage.view,
                image_layout: vk::ImageLayout::GENERAL,
                ..Default::default()
            })
            .collect();
        let sdf_material_id_infos: Vec<_> = render_targets
            .iter()
            .map(|targets| vk::DescriptorImageInfo {
                image_view: targets.sdf_material_id.view,
                image_layout: vk::ImageLayout::GENERAL,
                ..Default::default()
            })
            .collect();
        let resolved_primary_normal_depth_infos: Vec<_> = render_targets
            .iter()
            .map(|targets| vk::DescriptorImageInfo {
                image_view: targets.resolved_primary_normal_depth.view,
                image_layout: vk::ImageLayout::GENERAL,
                ..Default::default()
            })
            .collect();
        let resolved_primary_albedo_coverage_infos: Vec<_> = render_targets
            .iter()
            .map(|targets| vk::DescriptorImageInfo {
                image_view: targets.resolved_primary_albedo_coverage.view,
                image_layout: vk::ImageLayout::GENERAL,
                ..Default::default()
            })
            .collect();
        let resolved_secondary_normal_depth_infos: Vec<_> = render_targets
            .iter()
            .map(|targets| vk::DescriptorImageInfo {
                image_view: targets.resolved_secondary_normal_depth.view,
                image_layout: vk::ImageLayout::GENERAL,
                ..Default::default()
            })
            .collect();
        let resolved_secondary_albedo_coverage_infos: Vec<_> = render_targets
            .iter()
            .map(|targets| vk::DescriptorImageInfo {
                image_view: targets.resolved_secondary_albedo_coverage.view,
                image_layout: vk::ImageLayout::GENERAL,
                ..Default::default()
            })
            .collect();
        let resolved_material_ids_infos: Vec<_> = render_targets
            .iter()
            .map(|targets| vk::DescriptorImageInfo {
                image_view: targets.resolved_material_ids.view,
                image_layout: vk::ImageLayout::GENERAL,
                ..Default::default()
            })
            .collect();
        let surface_hdr_infos: Vec<_> = render_targets
            .iter()
            .map(|targets| vk::DescriptorImageInfo {
                image_view: targets.surface_hdr.view,
                image_layout: vk::ImageLayout::GENERAL,
                ..Default::default()
            })
            .collect();
        let overlay_hdr_infos: Vec<_> = render_targets
            .iter()
            .map(|targets| vk::DescriptorImageInfo {
                image_view: targets.overlay_hdr.view,
                image_layout: vk::ImageLayout::GENERAL,
                ..Default::default()
            })
            .collect();
        let raster_normal_depth_infos: Vec<_> = render_targets
            .iter()
            .map(|_| vk::DescriptorImageInfo {
                image_view: raster_normal_depth.view,
                image_layout: vk::ImageLayout::GENERAL,
                ..Default::default()
            })
            .collect();
        let raster_albedo_infos: Vec<_> = render_targets
            .iter()
            .map(|_| vk::DescriptorImageInfo {
                image_view: raster_albedo.view,
                image_layout: vk::ImageLayout::GENERAL,
                ..Default::default()
            })
            .collect();
        let raster_material_id_infos: Vec<_> = render_targets
            .iter()
            .map(|_| vk::DescriptorImageInfo {
                image_view: raster_material_id.view,
                image_layout: vk::ImageLayout::GENERAL,
                ..Default::default()
            })
            .collect();
        let sdf_depth_infos: Vec<_> = render_targets
            .iter()
            .map(|targets| vk::DescriptorImageInfo {
                image_view: targets.sdf_depth.view,
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
        let scene_color_infos: Vec<_> = render_targets
            .iter()
            .map(|targets| vk::DescriptorImageInfo {
                image_view: targets.scene_color.view,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                ..Default::default()
            })
            .collect();
        let transparent_back_depth_infos: Vec<_> = render_targets
            .iter()
            .map(|targets| vk::DescriptorImageInfo {
                image_view: targets.transparent_back_depth.view,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                ..Default::default()
            })
            .collect();
        let environment_image_info = vk::DescriptorImageInfo {
            image_view: resources.environment_map.view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            ..Default::default()
        };
        let environment_sampler_info = vk::DescriptorImageInfo {
            sampler: resources.environment_sampler,
            ..Default::default()
        };
        let bloom_ping_infos: Vec<_> = render_targets
            .iter()
            .map(|targets| vk::DescriptorImageInfo {
                image_view: targets.bloom_ping.view,
                image_layout: vk::ImageLayout::GENERAL,
                ..Default::default()
            })
            .collect();
        let bloom_pong_infos: Vec<_> = render_targets
            .iter()
            .map(|targets| vk::DescriptorImageInfo {
                image_view: targets.bloom_pong.view,
                image_layout: vk::ImageLayout::GENERAL,
                ..Default::default()
            })
            .collect();
        let camera_buffer_info = vk::DescriptorBufferInfo {
            buffer: resources.camera_buffer.vk_buffer,
            offset: 0,
            range: resources.camera_buffer_stride,
        };
        let material_buffer_3d_info = vk::DescriptorBufferInfo {
            buffer: resources.material_buffer_3d.vk_buffer,
            offset: 0,
            range: resources.material_buffer_3d_stride,
        };
        let buffer_3d_info = vk::DescriptorBufferInfo {
            buffer: resources.primitive_buffer.vk_buffer,
            offset: 0,
            range: resources.primitive_buffer_stride,
        };
        let grid_buffer_3d_info = vk::DescriptorBufferInfo {
            buffer: resources.grid_buffer_3d.vk_buffer,
            offset: 0,
            range: resources.grid_buffer_3d_stride,
        };

        let camera_buffer_2d_info = vk::DescriptorBufferInfo {
            buffer: resources.camera_buffer_2d.vk_buffer,
            offset: 0,
            range: resources.camera_buffer_2d_stride,
        };
        let tone_map_factor_infos: Vec<_> = (0..RENDER_FRAME_COUNT)
            .map(|index| vk::DescriptorBufferInfo {
                buffer: resources.tone_map_factor_buffer.vk_buffer,
                offset: index as u64 * resources.tone_map_factor_stride,
                range: resources.tone_map_factor_stride,
            })
            .collect();

        let mut write_descriptor_sets = vec![vk::WriteDescriptorSet {
            s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
            dst_set: raster_descriptor_set_2d,
            dst_binding: 0,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
            descriptor_count: 1,
            p_buffer_info: &camera_buffer_2d_info,
            ..Default::default()
        }];
        for (index, targets) in render_targets.iter().enumerate() {
            write_descriptor_sets.extend([
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: targets.raster_descriptor_set,
                    dst_binding: 1,
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
                    descriptor_count: 1,
                    p_buffer_info: &camera_buffer_info,
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: targets.raster_descriptor_set,
                    dst_binding: 2,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER_DYNAMIC,
                    descriptor_count: 1,
                    p_buffer_info: &material_buffer_3d_info,
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: targets.raster_descriptor_set,
                    dst_binding: 3,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &scene_color_infos[index],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: targets.raster_descriptor_set,
                    dst_binding: 4,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &transparent_back_depth_infos[index],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: targets.raster_descriptor_set,
                    dst_binding: 5,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &environment_image_info,
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: targets.raster_descriptor_set,
                    dst_binding: 6,
                    descriptor_type: vk::DescriptorType::SAMPLER,
                    descriptor_count: 1,
                    p_image_info: &environment_sampler_info,
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: targets.raster_descriptor_set,
                    dst_binding: 7,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &sdf_depth_infos[index],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set: targets.grid_descriptor_set,
                    dst_binding: 0,
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
                    descriptor_count: 1,
                    p_buffer_info: &camera_buffer_info,
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set: targets.grid_descriptor_set,
                    dst_binding: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER_DYNAMIC,
                    descriptor_count: 1,
                    p_buffer_info: &grid_buffer_3d_info,
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set: targets.grid_descriptor_set,
                    dst_binding: 2,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &sdf_depth_infos[index],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: targets.compute_descriptor_set,
                    dst_binding: 0,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &sdf_normal_coverage_infos[index],
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
                    dst_set: targets.compute_descriptor_set,
                    dst_binding: 3,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &sdf_material_id_infos[index],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: targets.compute_descriptor_set,
                    dst_binding: 4,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &sdf_depth_infos[index],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: targets.surface_resolve_descriptor_set,
                    dst_binding: 0,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &resolved_primary_normal_depth_infos[index],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: targets.surface_resolve_descriptor_set,
                    dst_binding: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &resolved_primary_albedo_coverage_infos[index],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: targets.surface_resolve_descriptor_set,
                    dst_binding: 2,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &resolved_secondary_normal_depth_infos[index],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: targets.surface_resolve_descriptor_set,
                    dst_binding: 3,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &resolved_secondary_albedo_coverage_infos[index],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: targets.surface_resolve_descriptor_set,
                    dst_binding: 4,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &resolved_material_ids_infos[index],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: targets.surface_resolve_descriptor_set,
                    dst_binding: 5,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &sdf_normal_coverage_infos[index],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: targets.surface_resolve_descriptor_set,
                    dst_binding: 6,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &sdf_depth_infos[index],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: targets.surface_resolve_descriptor_set,
                    dst_binding: 7,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &sdf_material_id_infos[index],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: targets.surface_resolve_descriptor_set,
                    dst_binding: 8,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &raster_normal_depth_infos[index],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: targets.surface_resolve_descriptor_set,
                    dst_binding: 9,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &raster_albedo_infos[index],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: targets.surface_resolve_descriptor_set,
                    dst_binding: 10,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &raster_material_id_infos[index],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: targets.surface_resolve_descriptor_set,
                    dst_binding: 11,
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
                    descriptor_count: 1,
                    p_buffer_info: &camera_buffer_info,
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: targets.surface_resolve_descriptor_set,
                    dst_binding: 12,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER_DYNAMIC,
                    descriptor_count: 1,
                    p_buffer_info: &material_buffer_3d_info,
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: targets.surface_lighting_descriptor_set,
                    dst_binding: 0,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &surface_hdr_infos[index],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: targets.surface_lighting_descriptor_set,
                    dst_binding: 1,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &resolved_primary_normal_depth_infos[index],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: targets.surface_lighting_descriptor_set,
                    dst_binding: 2,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &resolved_primary_albedo_coverage_infos[index],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: targets.surface_lighting_descriptor_set,
                    dst_binding: 3,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &resolved_secondary_normal_depth_infos[index],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: targets.surface_lighting_descriptor_set,
                    dst_binding: 4,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &resolved_secondary_albedo_coverage_infos[index],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: targets.surface_lighting_descriptor_set,
                    dst_binding: 5,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &resolved_material_ids_infos[index],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: targets.surface_lighting_descriptor_set,
                    dst_binding: 6,
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
                    descriptor_count: 1,
                    p_buffer_info: &camera_buffer_info,
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: targets.surface_lighting_descriptor_set,
                    dst_binding: 7,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER_DYNAMIC,
                    descriptor_count: 1,
                    p_buffer_info: &material_buffer_3d_info,
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: targets.surface_lighting_descriptor_set,
                    dst_binding: 8,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &environment_image_info,
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: targets.surface_lighting_descriptor_set,
                    dst_binding: 9,
                    descriptor_type: vk::DescriptorType::SAMPLER,
                    descriptor_count: 1,
                    p_image_info: &environment_sampler_info,
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: targets.surface_composite_descriptor_set,
                    dst_binding: 0,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &resolved_image_infos[index],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: targets.surface_composite_descriptor_set,
                    dst_binding: 1,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &surface_hdr_infos[index],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: targets.surface_composite_descriptor_set,
                    dst_binding: 2,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &overlay_hdr_infos[index],
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
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: targets.composite_descriptor_set,
                    dst_binding: 2,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &bloom_ping_infos[index],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: targets.composite_descriptor_set,
                    dst_binding: 3,
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: 1,
                    p_buffer_info: &tone_map_factor_infos[index],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: targets.bloom_descriptor_sets[0],
                    dst_binding: 0,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &resolved_image_infos[index],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: targets.bloom_descriptor_sets[0],
                    dst_binding: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &bloom_ping_infos[index],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: targets.bloom_descriptor_sets[1],
                    dst_binding: 0,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &bloom_ping_infos[index],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: targets.bloom_descriptor_sets[1],
                    dst_binding: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &bloom_pong_infos[index],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: targets.bloom_descriptor_sets[2],
                    dst_binding: 0,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &bloom_pong_infos[index],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: targets.bloom_descriptor_sets[2],
                    dst_binding: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &bloom_ping_infos[index],
                    ..Default::default()
                },
            ]);
        }

        let nv12_constants_buffer_info = vk::DescriptorBufferInfo {
            buffer: resources.nv12_constants_buffer.vk_buffer,
            offset: 0,
            range: vk::WHOLE_SIZE,
        };

        let nv12_buffer_infos: Vec<_> = nv12_output_buffers
            .iter()
            .map(|buffer| vk::DescriptorBufferInfo {
                buffer: buffer.vk_buffer,
                offset: 0,
                range: vk::WHOLE_SIZE,
            })
            .collect();

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

        let yuv444p_buffer_infos: Vec<_> = yuv444p_output_buffers
            .iter()
            .map(|buffer| vk::DescriptorBufferInfo {
                buffer: buffer.vk_buffer,
                offset: 0,
                range: vk::WHOLE_SIZE,
            })
            .collect();

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
            resources
                .ctx
                .device
                .update_descriptor_sets(&write_descriptor_sets, &[]);
        }

        Self {
            descriptor_pool: Arc::clone(resources.descriptor_pool),
            width,
            height,
            has_raster_gbuffer: needs_raster_gbuffer,
            has_overlay_hdr: needs_overlay_hdr,
            render_targets,
            raster_normal_depth,
            raster_normal_depth_state: TrackedImageState::UNDEFINED,
            raster_albedo,
            raster_albedo_state: TrackedImageState::UNDEFINED,
            raster_material_id,
            raster_material_id_state: TrackedImageState::UNDEFINED,
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
            raster_descriptor_set_2d,
            padded_bytes_per_row,
        }
    }

    pub(super) fn ensure_frame_attachments(
        &mut self,
        ctx: &Arc<VulkanContext>,
        plan: RecordingPlan,
        msaa_samples: u32,
    ) {
        if plan.raster_uses_depth && self.msaa_depth_texture.is_none() {
            self.msaa_depth_texture = Some(Image::new(
                ctx,
                plan.width * plan.ssaa_factor,
                plan.height * plan.ssaa_factor,
                vk::Format::D32_SFLOAT,
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                vk::ImageAspectFlags::DEPTH,
                msaa_to_vk_sample_count(msaa_samples),
            ));
            self.msaa_depth_texture_state = TrackedImageState::UNDEFINED;
        }

        let raster_sample_count = msaa_to_vk_sample_count(msaa_samples);
        if plan.execution.runs_raster()
            && !plan.analytic_2d
            && self.msaa_texture.is_none()
            && raster_sample_count != vk::SampleCountFlags::TYPE_1
        {
            self.msaa_texture = Some(Image::new(
                ctx,
                plan.width * plan.ssaa_factor,
                plan.height * plan.ssaa_factor,
                vk::Format::R16G16B16A16_SFLOAT,
                vk::ImageUsageFlags::COLOR_ATTACHMENT,
                vk::ImageAspectFlags::COLOR,
                raster_sample_count,
            ));
            self.msaa_texture_state = TrackedImageState::UNDEFINED;
        }
    }
}

impl Drop for TargetCache {
    fn drop(&mut self) {
        let mut descriptor_sets = Vec::with_capacity(40);
        for targets in &self.render_targets {
            descriptor_sets.extend([
                targets.compute_descriptor_set,
                targets.surface_resolve_descriptor_set,
                targets.surface_lighting_descriptor_set,
                targets.surface_composite_descriptor_set,
                targets.raster_descriptor_set,
                targets.grid_descriptor_set,
                targets.composite_descriptor_set,
            ]);
            descriptor_sets.extend(targets.bloom_descriptor_sets);
        }
        descriptor_sets.push(self.raster_descriptor_set_2d);
        descriptor_sets.extend(self.nv12_descriptor_sets);
        descriptor_sets.extend(self.yuv444p_descriptor_sets);
        descriptor_sets.extend(self.video_nv12_slots.iter().map(|slot| slot.descriptor_set));
        self.descriptor_pool.free(&descriptor_sets);
    }
}
