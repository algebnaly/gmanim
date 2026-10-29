use ash::vk;

use std::sync::Arc;

use crate::vulkan::context::VulkanContext;

use super::frame::{RENDER_FRAME_COUNT, TrackedImageState};
use super::prepared_frame::FrameRequirements;
use super::record::graph::{GraphImageId, SHARED_IMAGE_COUNT, SLOT_IMAGE_COUNT};
use super::video_output::{VIDEO_NV12_IMAGE_COUNT, VideoNv12Slot};
use super::{Buffer, DescriptorPool, Image, Nv12Constants, PipelineSet, msaa_to_vk_sample_count};

pub(super) struct TargetCacheResources<'a> {
    pub(super) ctx: &'a Arc<VulkanContext>,
    pub(super) descriptor_pool: &'a Arc<DescriptorPool>,
    pub(super) pipelines: &'a PipelineSet,
    pub(super) msaa_samples: u32,
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
    pub(super) sdf_normal_coverage: Image,
    pub(super) sdf_material_id: Image,
    pub(super) sdf_depth: Image,
    pub(super) resolved_primary_normal_depth: Image,
    pub(super) resolved_primary_albedo_coverage: Image,
    pub(super) resolved_secondary_normal_depth: Image,
    pub(super) resolved_secondary_albedo_coverage: Image,
    pub(super) resolved_material_ids: Image,
    pub(super) surface_hdr: Image,
    pub(super) overlay_hdr: Image,
    pub(super) resolved_texture: Image,
    pub(super) scene_color: Image,
    pub(super) transparent_back_depth: Image,
    pub(super) bloom_ping: Image,
    pub(super) bloom_pong: Image,
    pub(super) image_states: [TrackedImageState; SLOT_IMAGE_COUNT],
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

impl RenderTargetSet {
    pub(super) fn image(&self, id: GraphImageId) -> &Image {
        match id {
            GraphImageId::OutputRgba => &self.texture,
            GraphImageId::SdfNormalCoverage => &self.sdf_normal_coverage,
            GraphImageId::SdfMaterialId => &self.sdf_material_id,
            GraphImageId::SdfDepth => &self.sdf_depth,
            GraphImageId::ResolvedPrimaryNormalDepth => &self.resolved_primary_normal_depth,
            GraphImageId::ResolvedPrimaryAlbedoCoverage => &self.resolved_primary_albedo_coverage,
            GraphImageId::ResolvedSecondaryNormalDepth => &self.resolved_secondary_normal_depth,
            GraphImageId::ResolvedSecondaryAlbedoCoverage => {
                &self.resolved_secondary_albedo_coverage
            }
            GraphImageId::ResolvedMaterialIds => &self.resolved_material_ids,
            GraphImageId::SurfaceHdr => &self.surface_hdr,
            GraphImageId::OverlayHdr => &self.overlay_hdr,
            GraphImageId::ResolvedColor => &self.resolved_texture,
            GraphImageId::SceneColor => &self.scene_color,
            GraphImageId::TransparentBackDepth => &self.transparent_back_depth,
            GraphImageId::BloomPing => &self.bloom_ping,
            GraphImageId::BloomPong => &self.bloom_pong,
            _ => unreachable!("non-slot GraphImageId passed to RenderTargetSet::image"),
        }
    }
}

pub(super) struct TargetCache {
    descriptor_pool: Arc<DescriptorPool>,
    pub width: u32,
    pub height: u32,
    pub(super) requirements: FrameRequirements,
    pub(super) render_targets: [RenderTargetSet; RENDER_FRAME_COUNT],
    pub raster_normal_depth: Image,
    pub raster_albedo: Image,
    pub raster_material_id: Image,
    pub msaa_texture: Option<Image>,
    pub msaa_depth_texture: Option<Image>,
    pub(super) shared_image_states: [TrackedImageState; SHARED_IMAGE_COUNT],
    pub output_buffers: Vec<Buffer>,
    pub nv12_output_buffers: Vec<Buffer>,
    pub nv12_descriptor_sets: Vec<vk::DescriptorSet>,
    pub yuv444p_output_buffers: Vec<Buffer>,
    pub yuv444p_descriptor_sets: Vec<vk::DescriptorSet>,
    pub video_nv12_slots: Vec<VideoNv12Slot>,
    pub current_frame: usize,
    pub raster_descriptor_set_2d: vk::DescriptorSet,
    pub padded_bytes_per_row: u32,
}

#[derive(Clone, Copy)]
struct TargetExtents {
    rgba: (u32, u32),
    raster: (u32, u32),
    raster_gbuffer: (u32, u32),
    sdf: (u32, u32),
    surface: (u32, u32),
    overlay: (u32, u32),
    transparent: (u32, u32),
    bloom: (u32, u32),
}

impl TargetExtents {
    fn for_requirements(requirements: FrameRequirements) -> Self {
        // Fixed shader interfaces keep format-correct 1x1 placeholders for inactive inputs.
        // Only resources actually used by the plan receive the raster extent.
        let raster_extent = |enabled: bool| {
            if enabled {
                (requirements.raster_width, requirements.raster_height)
            } else {
                (1, 1)
            }
        };
        Self {
            rgba: if requirements.rgba_image {
                (requirements.width, requirements.height)
            } else {
                (1, 1)
            },
            raster: (requirements.raster_width, requirements.raster_height),
            raster_gbuffer: raster_extent(requirements.raster_gbuffer),
            sdf: raster_extent(requirements.sdf),
            surface: raster_extent(requirements.surface),
            overlay: raster_extent(requirements.overlay_hdr),
            transparent: raster_extent(requirements.transparent),
            bloom: if requirements.bloom {
                (
                    (requirements.width / 2).max(1),
                    (requirements.height / 2).max(1),
                )
            } else {
                (1, 1)
            },
        }
    }
}

fn recreate_image_if_needed(
    ctx: &Arc<VulkanContext>,
    image: &mut Image,
    state: &mut TrackedImageState,
    extent: (u32, u32),
    format: vk::Format,
    usage: vk::ImageUsageFlags,
    aspect: vk::ImageAspectFlags,
    samples: vk::SampleCountFlags,
) -> bool {
    if (image.width, image.height) == extent {
        return false;
    }
    *image = Image::new(ctx, extent.0, extent.1, format, usage, aspect, samples);
    *state = TrackedImageState::UNDEFINED;
    true
}

fn allocate_descriptor_sets(
    ctx: &VulkanContext,
    descriptor_pool: &DescriptorPool,
    layout: vk::DescriptorSetLayout,
    count: usize,
) -> Vec<vk::DescriptorSet> {
    if count == 0 {
        return Vec::new();
    }
    let layouts = vec![layout; count];
    unsafe {
        ctx.device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool.handle())
                    .set_layouts(&layouts),
            )
            .unwrap()
    }
}

fn create_readback_buffers(
    ctx: &Arc<VulkanContext>,
    enabled: bool,
    size: u64,
    usage: vk::BufferUsageFlags,
) -> Vec<Buffer> {
    (0..if enabled { RENDER_FRAME_COUNT } else { 0 })
        .map(|_| Buffer::new(ctx, size, usage, gpu_allocator::MemoryLocation::GpuToCpu))
        .collect()
}

const TEXTURE_USAGE: vk::ImageUsageFlags = vk::ImageUsageFlags::from_raw(
    vk::ImageUsageFlags::STORAGE.as_raw()
        | vk::ImageUsageFlags::SAMPLED.as_raw()
        | vk::ImageUsageFlags::TRANSFER_SRC.as_raw()
        | vk::ImageUsageFlags::TRANSFER_DST.as_raw()
        | vk::ImageUsageFlags::COLOR_ATTACHMENT.as_raw(),
);
const STORAGE_SAMPLED_USAGE: vk::ImageUsageFlags = vk::ImageUsageFlags::from_raw(
    vk::ImageUsageFlags::STORAGE.as_raw() | vk::ImageUsageFlags::SAMPLED.as_raw(),
);
const HDR_RESOLVE_USAGE: vk::ImageUsageFlags = vk::ImageUsageFlags::from_raw(
    vk::ImageUsageFlags::COLOR_ATTACHMENT.as_raw()
        | vk::ImageUsageFlags::SAMPLED.as_raw()
        | vk::ImageUsageFlags::TRANSFER_SRC.as_raw()
        | vk::ImageUsageFlags::STORAGE.as_raw(),
);
const COLOR_SAMPLED_USAGE: vk::ImageUsageFlags = vk::ImageUsageFlags::from_raw(
    vk::ImageUsageFlags::COLOR_ATTACHMENT.as_raw() | vk::ImageUsageFlags::SAMPLED.as_raw(),
);
const SCENE_COLOR_USAGE: vk::ImageUsageFlags = vk::ImageUsageFlags::from_raw(
    vk::ImageUsageFlags::SAMPLED.as_raw() | vk::ImageUsageFlags::TRANSFER_DST.as_raw(),
);
const BLOOM_USAGE: vk::ImageUsageFlags = vk::ImageUsageFlags::from_raw(
    vk::ImageUsageFlags::STORAGE.as_raw()
        | vk::ImageUsageFlags::SAMPLED.as_raw()
        | vk::ImageUsageFlags::TRANSFER_DST.as_raw(),
);

impl TargetCache {
    pub(super) fn satisfies(&self, requirements: FrameRequirements) -> bool {
        self.requirements == requirements
    }

    pub(super) fn vk_image(
        &self,
        frame_index: usize,
        video_frame_index: usize,
        id: GraphImageId,
    ) -> Option<vk::Image> {
        if id.slot_index().is_some() {
            Some(self.render_targets[frame_index].image(id).vk_image)
        } else {
            match id {
                GraphImageId::RasterNormalDepth => Some(self.raster_normal_depth.vk_image),
                GraphImageId::RasterAlbedo => Some(self.raster_albedo.vk_image),
                GraphImageId::RasterMaterialId => Some(self.raster_material_id.vk_image),
                GraphImageId::MsaaColor => self.msaa_texture.as_ref().map(|img| img.vk_image),
                GraphImageId::MsaaDepth => self.msaa_depth_texture.as_ref().map(|img| img.vk_image),
                GraphImageId::VideoNv12 => self
                    .video_nv12_slots
                    .get(video_frame_index)
                    .map(|slot| slot.image.vk_image),
                _ => None,
            }
        }
    }

    pub(super) fn new(
        requirements: FrameRequirements,
        resources: &TargetCacheResources<'_>,
    ) -> Self {
        let width = requirements.width;
        let height = requirements.height;
        let padded_bytes_per_row = requirements.padded_rgba_row_bytes;
        let extents = TargetExtents::for_requirements(requirements);

        resources.nv12_constants_buffer.write_bytes(
            0,
            bytemuck::bytes_of(&Nv12Constants {
                width,
                height,
                _padding: [0; 2],
            }),
        );

        let raster_sample_count = msaa_to_vk_sample_count(resources.msaa_samples);
        let raster_normal_depth = Image::new(
            resources.ctx,
            extents.raster_gbuffer.0,
            extents.raster_gbuffer.1,
            vk::Format::R16G16B16A16_SFLOAT,
            COLOR_SAMPLED_USAGE,
            vk::ImageAspectFlags::COLOR,
            raster_sample_count,
        );
        let raster_albedo = Image::new(
            resources.ctx,
            extents.raster_gbuffer.0,
            extents.raster_gbuffer.1,
            vk::Format::R16G16B16A16_SFLOAT,
            COLOR_SAMPLED_USAGE,
            vk::ImageAspectFlags::COLOR,
            raster_sample_count,
        );
        let raster_material_id = Image::new(
            resources.ctx,
            extents.raster_gbuffer.0,
            extents.raster_gbuffer.1,
            vk::Format::R16_UINT,
            COLOR_SAMPLED_USAGE,
            vk::ImageAspectFlags::COLOR,
            raster_sample_count,
        );

        let mut render_targets = std::array::from_fn(|_| {
            let color_image = |extent: (u32, u32), format, usage| {
                Image::new(
                    resources.ctx,
                    extent.0,
                    extent.1,
                    format,
                    usage,
                    vk::ImageAspectFlags::COLOR,
                    vk::SampleCountFlags::TYPE_1,
                )
            };
            RenderTargetSet {
                texture: color_image(extents.rgba, vk::Format::R8G8B8A8_UNORM, TEXTURE_USAGE),
                sdf_normal_coverage: color_image(
                    extents.sdf,
                    vk::Format::R16G16B16A16_SFLOAT,
                    STORAGE_SAMPLED_USAGE,
                ),
                sdf_material_id: color_image(
                    extents.sdf,
                    vk::Format::R32_UINT,
                    STORAGE_SAMPLED_USAGE,
                ),
                sdf_depth: color_image(extents.sdf, vk::Format::R32_SFLOAT, STORAGE_SAMPLED_USAGE),
                resolved_primary_normal_depth: color_image(
                    extents.surface,
                    vk::Format::R16G16B16A16_SFLOAT,
                    STORAGE_SAMPLED_USAGE,
                ),
                resolved_primary_albedo_coverage: color_image(
                    extents.surface,
                    vk::Format::R16G16B16A16_SFLOAT,
                    STORAGE_SAMPLED_USAGE,
                ),
                resolved_secondary_normal_depth: color_image(
                    extents.surface,
                    vk::Format::R16G16B16A16_SFLOAT,
                    STORAGE_SAMPLED_USAGE,
                ),
                resolved_secondary_albedo_coverage: color_image(
                    extents.surface,
                    vk::Format::R16G16B16A16_SFLOAT,
                    STORAGE_SAMPLED_USAGE,
                ),
                resolved_material_ids: color_image(
                    extents.surface,
                    vk::Format::R32_UINT,
                    STORAGE_SAMPLED_USAGE,
                ),
                surface_hdr: color_image(
                    extents.surface,
                    vk::Format::R16G16B16A16_SFLOAT,
                    HDR_RESOLVE_USAGE,
                ),
                overlay_hdr: color_image(
                    extents.overlay,
                    vk::Format::R16G16B16A16_SFLOAT,
                    COLOR_SAMPLED_USAGE,
                ),
                resolved_texture: color_image(
                    extents.raster,
                    vk::Format::R16G16B16A16_SFLOAT,
                    HDR_RESOLVE_USAGE,
                ),
                scene_color: color_image(
                    extents.transparent,
                    vk::Format::R16G16B16A16_SFLOAT,
                    SCENE_COLOR_USAGE,
                ),
                transparent_back_depth: color_image(
                    extents.transparent,
                    vk::Format::R32_SFLOAT,
                    COLOR_SAMPLED_USAGE,
                ),
                bloom_ping: color_image(
                    extents.bloom,
                    vk::Format::R16G16B16A16_SFLOAT,
                    BLOOM_USAGE,
                ),
                bloom_pong: color_image(
                    extents.bloom,
                    vk::Format::R16G16B16A16_SFLOAT,
                    BLOOM_USAGE,
                ),
                image_states: [TrackedImageState::UNDEFINED; SLOT_IMAGE_COUNT],
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

        let compute_sets = allocate_descriptor_sets(
            resources.ctx,
            resources.descriptor_pool,
            resources.pipelines.compute_descriptor_set_layout,
            RENDER_FRAME_COUNT,
        );
        let surface_resolve_sets = allocate_descriptor_sets(
            resources.ctx,
            resources.descriptor_pool,
            resources.pipelines.surface_resolve_descriptor_set_layout,
            RENDER_FRAME_COUNT,
        );
        let surface_lighting_sets = allocate_descriptor_sets(
            resources.ctx,
            resources.descriptor_pool,
            resources.pipelines.surface_lighting_descriptor_set_layout,
            RENDER_FRAME_COUNT,
        );
        let surface_composite_sets = allocate_descriptor_sets(
            resources.ctx,
            resources.descriptor_pool,
            resources.pipelines.surface_composite_descriptor_set_layout,
            RENDER_FRAME_COUNT,
        );
        let raster_sets = allocate_descriptor_sets(
            resources.ctx,
            resources.descriptor_pool,
            resources.pipelines.raster_descriptor_set_layout,
            RENDER_FRAME_COUNT,
        );
        let grid_sets = allocate_descriptor_sets(
            resources.ctx,
            resources.descriptor_pool,
            resources.pipelines.grid_descriptor_set_layout,
            RENDER_FRAME_COUNT,
        );
        let composite_sets = allocate_descriptor_sets(
            resources.ctx,
            resources.descriptor_pool,
            resources.pipelines.composite_descriptor_set_layout,
            RENDER_FRAME_COUNT,
        );
        let bloom_sets = allocate_descriptor_sets(
            resources.ctx,
            resources.descriptor_pool,
            resources.pipelines.bloom_descriptor_set_layout,
            RENDER_FRAME_COUNT * 3,
        );
        let (bloom_chunks, remainder) = bloom_sets.as_chunks::<3>();
        debug_assert!(remainder.is_empty());

        for index in 0..RENDER_FRAME_COUNT {
            let targets = &mut render_targets[index];
            targets.compute_descriptor_set = compute_sets[index];
            targets.surface_resolve_descriptor_set = surface_resolve_sets[index];
            targets.surface_lighting_descriptor_set = surface_lighting_sets[index];
            targets.surface_composite_descriptor_set = surface_composite_sets[index];
            targets.raster_descriptor_set = raster_sets[index];
            targets.grid_descriptor_set = grid_sets[index];
            targets.composite_descriptor_set = composite_sets[index];
            targets.bloom_descriptor_sets = bloom_chunks[index];
        }

        let raster_descriptor_set_2d = allocate_descriptor_sets(
            resources.ctx,
            resources.descriptor_pool,
            resources.pipelines.raster_descriptor_set_layout_2d,
            1,
        )[0];

        let output_buffers = create_readback_buffers(
            resources.ctx,
            requirements.outputs.cpu_rgba,
            u64::from(padded_bytes_per_row) * u64::from(height),
            vk::BufferUsageFlags::TRANSFER_DST,
        );
        let nv12_output_buffers = create_readback_buffers(
            resources.ctx,
            requirements.outputs.cpu_nv12,
            u64::from(width) * u64::from(height) * 3 / 2,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        );
        let yuv444p_output_buffers = create_readback_buffers(
            resources.ctx,
            requirements.outputs.cpu_yuv444p,
            u64::from(width) * u64::from(height) * 3,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        );

        let nv12_descriptor_sets = allocate_descriptor_sets(
            resources.ctx,
            resources.descriptor_pool,
            resources.pipelines.nv12_descriptor_set_layout,
            nv12_output_buffers.len(),
        );
        let yuv444p_descriptor_sets = allocate_descriptor_sets(
            resources.ctx,
            resources.descriptor_pool,
            resources.pipelines.nv12_descriptor_set_layout,
            yuv444p_output_buffers.len(),
        );

        let mut video_nv12_slots = (0..if requirements.outputs.vulkan_video {
            VIDEO_NV12_IMAGE_COUNT
        } else {
            0
        })
            .map(|_| VideoNv12Slot::new(resources.ctx, width, height))
            .collect::<Vec<_>>();
        let video_nv12_descriptor_sets = allocate_descriptor_sets(
            resources.ctx,
            resources.descriptor_pool,
            resources.pipelines.video_nv12_descriptor_set_layout,
            video_nv12_slots.len(),
        );
        for (slot, descriptor_set) in video_nv12_slots.iter_mut().zip(video_nv12_descriptor_sets) {
            slot.descriptor_set = descriptor_set;
        }

        let cache = Self {
            descriptor_pool: Arc::clone(resources.descriptor_pool),
            width,
            height,
            requirements,
            render_targets,
            raster_normal_depth,
            raster_albedo,
            raster_material_id,
            msaa_texture: None,
            msaa_depth_texture: None,
            shared_image_states: [TrackedImageState::UNDEFINED; SHARED_IMAGE_COUNT],
            output_buffers,
            nv12_output_buffers,
            nv12_descriptor_sets,
            yuv444p_output_buffers,
            yuv444p_descriptor_sets,
            video_nv12_slots,
            current_frame: 0,
            raster_descriptor_set_2d,
            padded_bytes_per_row,
        };
        cache.write_descriptors(resources);
        cache
    }

    /// Incrementally reconfigures only the render targets and readback buffers whose
    /// required extents or active status changed, preserving unaffected GPU images,
    /// tracked layouts, and core descriptor sets.
    pub(super) fn reconfigure(
        &mut self,
        requirements: FrameRequirements,
        resources: &TargetCacheResources<'_>,
    ) {
        if self.satisfies(requirements) {
            return;
        }

        unsafe {
            // Only the graphics/compute queue can still reference render targets.
            // Do not wait for the video encode queue here.
            resources
                .ctx
                .device
                .queue_wait_idle(resources.ctx.queue)
                .unwrap();
        }

        let width = requirements.width;
        let height = requirements.height;
        let padded_bytes_per_row = requirements.padded_rgba_row_bytes;
        let dims_changed = (self.width, self.height) != (width, height);
        let extents = TargetExtents::for_requirements(requirements);
        let raster_sample_count = msaa_to_vk_sample_count(resources.msaa_samples);

        if dims_changed {
            resources.nv12_constants_buffer.write_bytes(
                0,
                bytemuck::bytes_of(&Nv12Constants {
                    width,
                    height,
                    _padding: [0; 2],
                }),
            );
        }

        recreate_image_if_needed(
            resources.ctx,
            &mut self.raster_normal_depth,
            &mut self.shared_image_states[GraphImageId::RasterNormalDepth.shared_index().unwrap()],
            extents.raster_gbuffer,
            vk::Format::R16G16B16A16_SFLOAT,
            COLOR_SAMPLED_USAGE,
            vk::ImageAspectFlags::COLOR,
            raster_sample_count,
        );
        recreate_image_if_needed(
            resources.ctx,
            &mut self.raster_albedo,
            &mut self.shared_image_states[GraphImageId::RasterAlbedo.shared_index().unwrap()],
            extents.raster_gbuffer,
            vk::Format::R16G16B16A16_SFLOAT,
            COLOR_SAMPLED_USAGE,
            vk::ImageAspectFlags::COLOR,
            raster_sample_count,
        );
        recreate_image_if_needed(
            resources.ctx,
            &mut self.raster_material_id,
            &mut self.shared_image_states[GraphImageId::RasterMaterialId.shared_index().unwrap()],
            extents.raster_gbuffer,
            vk::Format::R16_UINT,
            COLOR_SAMPLED_USAGE,
            vk::ImageAspectFlags::COLOR,
            raster_sample_count,
        );

        if !requirements.depth
            || self
                .msaa_depth_texture
                .as_ref()
                .is_some_and(|tex| (tex.width, tex.height) != extents.raster)
        {
            self.msaa_depth_texture = None;
            self.shared_image_states[GraphImageId::MsaaDepth.shared_index().unwrap()] =
                TrackedImageState::UNDEFINED;
        }

        if !requirements.multisampled_raster
            || self
                .msaa_texture
                .as_ref()
                .is_some_and(|tex| (tex.width, tex.height) != extents.raster)
        {
            self.msaa_texture = None;
            self.shared_image_states[GraphImageId::MsaaColor.shared_index().unwrap()] =
                TrackedImageState::UNDEFINED;
        }

        for targets in &mut self.render_targets {
            recreate_image_if_needed(
                resources.ctx,
                &mut targets.texture,
                &mut targets.image_states[GraphImageId::OutputRgba.index()],
                extents.rgba,
                vk::Format::R8G8B8A8_UNORM,
                TEXTURE_USAGE,
                vk::ImageAspectFlags::COLOR,
                vk::SampleCountFlags::TYPE_1,
            );
            recreate_image_if_needed(
                resources.ctx,
                &mut targets.sdf_normal_coverage,
                &mut targets.image_states[GraphImageId::SdfNormalCoverage.index()],
                extents.sdf,
                vk::Format::R16G16B16A16_SFLOAT,
                STORAGE_SAMPLED_USAGE,
                vk::ImageAspectFlags::COLOR,
                vk::SampleCountFlags::TYPE_1,
            );
            recreate_image_if_needed(
                resources.ctx,
                &mut targets.sdf_material_id,
                &mut targets.image_states[GraphImageId::SdfMaterialId.index()],
                extents.sdf,
                vk::Format::R32_UINT,
                STORAGE_SAMPLED_USAGE,
                vk::ImageAspectFlags::COLOR,
                vk::SampleCountFlags::TYPE_1,
            );
            recreate_image_if_needed(
                resources.ctx,
                &mut targets.sdf_depth,
                &mut targets.image_states[GraphImageId::SdfDepth.index()],
                extents.sdf,
                vk::Format::R32_SFLOAT,
                STORAGE_SAMPLED_USAGE,
                vk::ImageAspectFlags::COLOR,
                vk::SampleCountFlags::TYPE_1,
            );
            recreate_image_if_needed(
                resources.ctx,
                &mut targets.resolved_primary_normal_depth,
                &mut targets.image_states[GraphImageId::ResolvedPrimaryNormalDepth.index()],
                extents.surface,
                vk::Format::R16G16B16A16_SFLOAT,
                STORAGE_SAMPLED_USAGE,
                vk::ImageAspectFlags::COLOR,
                vk::SampleCountFlags::TYPE_1,
            );
            recreate_image_if_needed(
                resources.ctx,
                &mut targets.resolved_primary_albedo_coverage,
                &mut targets.image_states[GraphImageId::ResolvedPrimaryAlbedoCoverage.index()],
                extents.surface,
                vk::Format::R16G16B16A16_SFLOAT,
                STORAGE_SAMPLED_USAGE,
                vk::ImageAspectFlags::COLOR,
                vk::SampleCountFlags::TYPE_1,
            );
            recreate_image_if_needed(
                resources.ctx,
                &mut targets.resolved_secondary_normal_depth,
                &mut targets.image_states[GraphImageId::ResolvedSecondaryNormalDepth.index()],
                extents.surface,
                vk::Format::R16G16B16A16_SFLOAT,
                STORAGE_SAMPLED_USAGE,
                vk::ImageAspectFlags::COLOR,
                vk::SampleCountFlags::TYPE_1,
            );
            recreate_image_if_needed(
                resources.ctx,
                &mut targets.resolved_secondary_albedo_coverage,
                &mut targets.image_states[GraphImageId::ResolvedSecondaryAlbedoCoverage.index()],
                extents.surface,
                vk::Format::R16G16B16A16_SFLOAT,
                STORAGE_SAMPLED_USAGE,
                vk::ImageAspectFlags::COLOR,
                vk::SampleCountFlags::TYPE_1,
            );
            recreate_image_if_needed(
                resources.ctx,
                &mut targets.resolved_material_ids,
                &mut targets.image_states[GraphImageId::ResolvedMaterialIds.index()],
                extents.surface,
                vk::Format::R32_UINT,
                STORAGE_SAMPLED_USAGE,
                vk::ImageAspectFlags::COLOR,
                vk::SampleCountFlags::TYPE_1,
            );
            recreate_image_if_needed(
                resources.ctx,
                &mut targets.surface_hdr,
                &mut targets.image_states[GraphImageId::SurfaceHdr.index()],
                extents.surface,
                vk::Format::R16G16B16A16_SFLOAT,
                HDR_RESOLVE_USAGE,
                vk::ImageAspectFlags::COLOR,
                vk::SampleCountFlags::TYPE_1,
            );
            recreate_image_if_needed(
                resources.ctx,
                &mut targets.overlay_hdr,
                &mut targets.image_states[GraphImageId::OverlayHdr.index()],
                extents.overlay,
                vk::Format::R16G16B16A16_SFLOAT,
                COLOR_SAMPLED_USAGE,
                vk::ImageAspectFlags::COLOR,
                vk::SampleCountFlags::TYPE_1,
            );
            recreate_image_if_needed(
                resources.ctx,
                &mut targets.resolved_texture,
                &mut targets.image_states[GraphImageId::ResolvedColor.index()],
                extents.raster,
                vk::Format::R16G16B16A16_SFLOAT,
                HDR_RESOLVE_USAGE,
                vk::ImageAspectFlags::COLOR,
                vk::SampleCountFlags::TYPE_1,
            );
            recreate_image_if_needed(
                resources.ctx,
                &mut targets.scene_color,
                &mut targets.image_states[GraphImageId::SceneColor.index()],
                extents.transparent,
                vk::Format::R16G16B16A16_SFLOAT,
                SCENE_COLOR_USAGE,
                vk::ImageAspectFlags::COLOR,
                vk::SampleCountFlags::TYPE_1,
            );
            recreate_image_if_needed(
                resources.ctx,
                &mut targets.transparent_back_depth,
                &mut targets.image_states[GraphImageId::TransparentBackDepth.index()],
                extents.transparent,
                vk::Format::R32_SFLOAT,
                COLOR_SAMPLED_USAGE,
                vk::ImageAspectFlags::COLOR,
                vk::SampleCountFlags::TYPE_1,
            );
            let ping_recreated = recreate_image_if_needed(
                resources.ctx,
                &mut targets.bloom_ping,
                &mut targets.image_states[GraphImageId::BloomPing.index()],
                extents.bloom,
                vk::Format::R16G16B16A16_SFLOAT,
                BLOOM_USAGE,
                vk::ImageAspectFlags::COLOR,
                vk::SampleCountFlags::TYPE_1,
            );
            let pong_recreated = recreate_image_if_needed(
                resources.ctx,
                &mut targets.bloom_pong,
                &mut targets.image_states[GraphImageId::BloomPong.index()],
                extents.bloom,
                vk::Format::R16G16B16A16_SFLOAT,
                BLOOM_USAGE,
                vk::ImageAspectFlags::COLOR,
                vk::SampleCountFlags::TYPE_1,
            );
            if ping_recreated || pong_recreated {
                targets.bloom_contains_data = false;
            }
        }

        // Reconfigure CPU RGBA readback buffers
        if !requirements.outputs.cpu_rgba {
            self.output_buffers.clear();
        } else if self.output_buffers.is_empty()
            || dims_changed
            || self.padded_bytes_per_row != padded_bytes_per_row
        {
            self.output_buffers = create_readback_buffers(
                resources.ctx,
                true,
                u64::from(padded_bytes_per_row) * u64::from(height),
                vk::BufferUsageFlags::TRANSFER_DST,
            );
        }

        // Reconfigure CPU NV12 readback buffers & descriptor sets
        if !requirements.outputs.cpu_nv12 {
            self.nv12_output_buffers.clear();
            if !self.nv12_descriptor_sets.is_empty() {
                self.descriptor_pool.free(&self.nv12_descriptor_sets);
                self.nv12_descriptor_sets.clear();
            }
        } else {
            if self.nv12_output_buffers.is_empty() || dims_changed {
                self.nv12_output_buffers = create_readback_buffers(
                    resources.ctx,
                    true,
                    u64::from(width) * u64::from(height) * 3 / 2,
                    vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                );
            }
            if self.nv12_descriptor_sets.is_empty() {
                self.nv12_descriptor_sets = allocate_descriptor_sets(
                    resources.ctx,
                    resources.descriptor_pool,
                    resources.pipelines.nv12_descriptor_set_layout,
                    RENDER_FRAME_COUNT,
                );
            }
        }

        // Reconfigure CPU YUV444P readback buffers & descriptor sets
        if !requirements.outputs.cpu_yuv444p {
            self.yuv444p_output_buffers.clear();
            if !self.yuv444p_descriptor_sets.is_empty() {
                self.descriptor_pool.free(&self.yuv444p_descriptor_sets);
                self.yuv444p_descriptor_sets.clear();
            }
        } else {
            if self.yuv444p_output_buffers.is_empty() || dims_changed {
                self.yuv444p_output_buffers = create_readback_buffers(
                    resources.ctx,
                    true,
                    u64::from(width) * u64::from(height) * 3,
                    vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                );
            }
            if self.yuv444p_descriptor_sets.is_empty() {
                self.yuv444p_descriptor_sets = allocate_descriptor_sets(
                    resources.ctx,
                    resources.descriptor_pool,
                    resources.pipelines.nv12_descriptor_set_layout,
                    RENDER_FRAME_COUNT,
                );
            }
        }

        // Reconfigure Vulkan Video NV12 slots
        if !requirements.outputs.vulkan_video {
            if !self.video_nv12_slots.is_empty() {
                let sets: Vec<_> = self
                    .video_nv12_slots
                    .iter()
                    .map(|slot| slot.descriptor_set)
                    .collect();
                self.descriptor_pool.free(&sets);
                self.video_nv12_slots.clear();
            }
        } else if self.video_nv12_slots.is_empty() || dims_changed {
            if !self.video_nv12_slots.is_empty() {
                let sets: Vec<_> = self
                    .video_nv12_slots
                    .iter()
                    .map(|slot| slot.descriptor_set)
                    .collect();
                self.descriptor_pool.free(&sets);
                self.video_nv12_slots.clear();
            }
            let mut video_nv12_slots = (0..VIDEO_NV12_IMAGE_COUNT)
                .map(|_| VideoNv12Slot::new(resources.ctx, width, height))
                .collect::<Vec<_>>();
            let video_nv12_descriptor_sets = allocate_descriptor_sets(
                resources.ctx,
                resources.descriptor_pool,
                resources.pipelines.video_nv12_descriptor_set_layout,
                video_nv12_slots.len(),
            );
            for (slot, descriptor_set) in
                video_nv12_slots.iter_mut().zip(video_nv12_descriptor_sets)
            {
                slot.descriptor_set = descriptor_set;
            }
            self.video_nv12_slots = video_nv12_slots;
        }

        self.width = width;
        self.height = height;
        self.padded_bytes_per_row = padded_bytes_per_row;
        self.requirements = requirements;
        self.write_descriptors(resources);
    }

    fn write_descriptors(&self, resources: &TargetCacheResources<'_>) {
        let general_info = |view| vk::DescriptorImageInfo {
            image_view: view,
            image_layout: vk::ImageLayout::GENERAL,
            ..Default::default()
        };
        let sampled_info = |view| vk::DescriptorImageInfo {
            image_view: view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            ..Default::default()
        };

        let image_infos: Vec<_> = self
            .render_targets
            .iter()
            .map(|t| general_info(t.texture.view))
            .collect();
        let sdf_normal_coverage_infos: Vec<_> = self
            .render_targets
            .iter()
            .map(|t| general_info(t.sdf_normal_coverage.view))
            .collect();
        let sdf_material_id_infos: Vec<_> = self
            .render_targets
            .iter()
            .map(|t| general_info(t.sdf_material_id.view))
            .collect();
        let sdf_depth_infos: Vec<_> = self
            .render_targets
            .iter()
            .map(|t| general_info(t.sdf_depth.view))
            .collect();
        let resolved_primary_normal_depth_infos: Vec<_> = self
            .render_targets
            .iter()
            .map(|t| general_info(t.resolved_primary_normal_depth.view))
            .collect();
        let resolved_primary_albedo_coverage_infos: Vec<_> = self
            .render_targets
            .iter()
            .map(|t| general_info(t.resolved_primary_albedo_coverage.view))
            .collect();
        let resolved_secondary_normal_depth_infos: Vec<_> = self
            .render_targets
            .iter()
            .map(|t| general_info(t.resolved_secondary_normal_depth.view))
            .collect();
        let resolved_secondary_albedo_coverage_infos: Vec<_> = self
            .render_targets
            .iter()
            .map(|t| general_info(t.resolved_secondary_albedo_coverage.view))
            .collect();
        let resolved_material_ids_infos: Vec<_> = self
            .render_targets
            .iter()
            .map(|t| general_info(t.resolved_material_ids.view))
            .collect();
        let surface_hdr_infos: Vec<_> = self
            .render_targets
            .iter()
            .map(|t| general_info(t.surface_hdr.view))
            .collect();
        let overlay_hdr_infos: Vec<_> = self
            .render_targets
            .iter()
            .map(|t| general_info(t.overlay_hdr.view))
            .collect();
        let resolved_image_infos: Vec<_> = self
            .render_targets
            .iter()
            .map(|t| general_info(t.resolved_texture.view))
            .collect();
        let scene_color_infos: Vec<_> = self
            .render_targets
            .iter()
            .map(|t| sampled_info(t.scene_color.view))
            .collect();
        let transparent_back_depth_infos: Vec<_> = self
            .render_targets
            .iter()
            .map(|t| sampled_info(t.transparent_back_depth.view))
            .collect();
        let bloom_ping_infos: Vec<_> = self
            .render_targets
            .iter()
            .map(|t| general_info(t.bloom_ping.view))
            .collect();
        let bloom_pong_infos: Vec<_> = self
            .render_targets
            .iter()
            .map(|t| general_info(t.bloom_pong.view))
            .collect();

        let raster_normal_depth_info = general_info(self.raster_normal_depth.view);
        let raster_albedo_info = general_info(self.raster_albedo.view);
        let raster_material_id_info = general_info(self.raster_material_id.view);
        let environment_image_info = sampled_info(resources.environment_map.view);
        let environment_sampler_info = vk::DescriptorImageInfo {
            sampler: resources.environment_sampler,
            ..Default::default()
        };

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
            dst_set: self.raster_descriptor_set_2d,
            dst_binding: 0,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
            descriptor_count: 1,
            p_buffer_info: &camera_buffer_2d_info,
            ..Default::default()
        }];

        for (index, targets) in self.render_targets.iter().enumerate() {
            write_descriptor_sets.extend([
                vk::WriteDescriptorSet {
                    dst_set: targets.raster_descriptor_set,
                    dst_binding: 1,
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
                    descriptor_count: 1,
                    p_buffer_info: &camera_buffer_info,
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set: targets.raster_descriptor_set,
                    dst_binding: 2,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER_DYNAMIC,
                    descriptor_count: 1,
                    p_buffer_info: &material_buffer_3d_info,
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set: targets.raster_descriptor_set,
                    dst_binding: 3,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &scene_color_infos[index],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set: targets.raster_descriptor_set,
                    dst_binding: 4,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &transparent_back_depth_infos[index],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set: targets.raster_descriptor_set,
                    dst_binding: 5,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &environment_image_info,
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set: targets.raster_descriptor_set,
                    dst_binding: 6,
                    descriptor_type: vk::DescriptorType::SAMPLER,
                    descriptor_count: 1,
                    p_image_info: &environment_sampler_info,
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
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
                    dst_set: targets.compute_descriptor_set,
                    dst_binding: 0,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &sdf_normal_coverage_infos[index],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set: targets.compute_descriptor_set,
                    dst_binding: 1,
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
                    descriptor_count: 1,
                    p_buffer_info: &camera_buffer_info,
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set: targets.compute_descriptor_set,
                    dst_binding: 2,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER_DYNAMIC,
                    descriptor_count: 1,
                    p_buffer_info: &buffer_3d_info,
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set: targets.compute_descriptor_set,
                    dst_binding: 3,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &sdf_material_id_infos[index],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set: targets.compute_descriptor_set,
                    dst_binding: 4,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &sdf_depth_infos[index],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set: targets.surface_resolve_descriptor_set,
                    dst_binding: 0,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &resolved_primary_normal_depth_infos[index],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set: targets.surface_resolve_descriptor_set,
                    dst_binding: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &resolved_primary_albedo_coverage_infos[index],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set: targets.surface_resolve_descriptor_set,
                    dst_binding: 2,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &resolved_secondary_normal_depth_infos[index],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set: targets.surface_resolve_descriptor_set,
                    dst_binding: 3,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &resolved_secondary_albedo_coverage_infos[index],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set: targets.surface_resolve_descriptor_set,
                    dst_binding: 4,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &resolved_material_ids_infos[index],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set: targets.surface_resolve_descriptor_set,
                    dst_binding: 5,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &sdf_normal_coverage_infos[index],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set: targets.surface_resolve_descriptor_set,
                    dst_binding: 6,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &sdf_depth_infos[index],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set: targets.surface_resolve_descriptor_set,
                    dst_binding: 7,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &sdf_material_id_infos[index],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set: targets.surface_resolve_descriptor_set,
                    dst_binding: 8,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &raster_normal_depth_info,
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set: targets.surface_resolve_descriptor_set,
                    dst_binding: 9,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &raster_albedo_info,
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set: targets.surface_resolve_descriptor_set,
                    dst_binding: 10,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &raster_material_id_info,
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set: targets.surface_resolve_descriptor_set,
                    dst_binding: 11,
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
                    descriptor_count: 1,
                    p_buffer_info: &camera_buffer_info,
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set: targets.surface_resolve_descriptor_set,
                    dst_binding: 12,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER_DYNAMIC,
                    descriptor_count: 1,
                    p_buffer_info: &material_buffer_3d_info,
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set: targets.surface_lighting_descriptor_set,
                    dst_binding: 0,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &surface_hdr_infos[index],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set: targets.surface_lighting_descriptor_set,
                    dst_binding: 1,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &resolved_primary_normal_depth_infos[index],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set: targets.surface_lighting_descriptor_set,
                    dst_binding: 2,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &resolved_primary_albedo_coverage_infos[index],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set: targets.surface_lighting_descriptor_set,
                    dst_binding: 3,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &resolved_secondary_normal_depth_infos[index],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set: targets.surface_lighting_descriptor_set,
                    dst_binding: 4,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &resolved_secondary_albedo_coverage_infos[index],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set: targets.surface_lighting_descriptor_set,
                    dst_binding: 5,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &resolved_material_ids_infos[index],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set: targets.surface_lighting_descriptor_set,
                    dst_binding: 6,
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
                    descriptor_count: 1,
                    p_buffer_info: &camera_buffer_info,
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set: targets.surface_lighting_descriptor_set,
                    dst_binding: 7,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER_DYNAMIC,
                    descriptor_count: 1,
                    p_buffer_info: &material_buffer_3d_info,
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set: targets.surface_lighting_descriptor_set,
                    dst_binding: 8,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &environment_image_info,
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set: targets.surface_lighting_descriptor_set,
                    dst_binding: 9,
                    descriptor_type: vk::DescriptorType::SAMPLER,
                    descriptor_count: 1,
                    p_image_info: &environment_sampler_info,
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set: targets.surface_composite_descriptor_set,
                    dst_binding: 0,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &resolved_image_infos[index],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set: targets.surface_composite_descriptor_set,
                    dst_binding: 1,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &surface_hdr_infos[index],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set: targets.surface_composite_descriptor_set,
                    dst_binding: 2,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &overlay_hdr_infos[index],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set: targets.composite_descriptor_set,
                    dst_binding: 0,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &image_infos[index],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set: targets.composite_descriptor_set,
                    dst_binding: 1,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &resolved_image_infos[index],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set: targets.composite_descriptor_set,
                    dst_binding: 2,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &bloom_ping_infos[index],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set: targets.composite_descriptor_set,
                    dst_binding: 3,
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: 1,
                    p_buffer_info: &tone_map_factor_infos[index],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set: targets.bloom_descriptor_sets[0],
                    dst_binding: 0,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &resolved_image_infos[index],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set: targets.bloom_descriptor_sets[0],
                    dst_binding: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &bloom_ping_infos[index],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set: targets.bloom_descriptor_sets[1],
                    dst_binding: 0,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &bloom_ping_infos[index],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set: targets.bloom_descriptor_sets[1],
                    dst_binding: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &bloom_pong_infos[index],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set: targets.bloom_descriptor_sets[2],
                    dst_binding: 0,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &bloom_pong_infos[index],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
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

        let nv12_buffer_infos: Vec<_> = self
            .nv12_output_buffers
            .iter()
            .map(|buffer| vk::DescriptorBufferInfo {
                buffer: buffer.vk_buffer,
                offset: 0,
                range: vk::WHOLE_SIZE,
            })
            .collect();
        for i in 0..self.nv12_descriptor_sets.len() {
            write_descriptor_sets.extend([
                vk::WriteDescriptorSet {
                    dst_set: self.nv12_descriptor_sets[i],
                    dst_binding: 0,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &image_infos[i],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set: self.nv12_descriptor_sets[i],
                    dst_binding: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: 1,
                    p_buffer_info: &nv12_buffer_infos[i],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set: self.nv12_descriptor_sets[i],
                    dst_binding: 2,
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: 1,
                    p_buffer_info: &nv12_constants_buffer_info,
                    ..Default::default()
                },
            ]);
        }

        let yuv444p_buffer_infos: Vec<_> = self
            .yuv444p_output_buffers
            .iter()
            .map(|buffer| vk::DescriptorBufferInfo {
                buffer: buffer.vk_buffer,
                offset: 0,
                range: vk::WHOLE_SIZE,
            })
            .collect();
        for i in 0..self.yuv444p_descriptor_sets.len() {
            write_descriptor_sets.extend([
                vk::WriteDescriptorSet {
                    dst_set: self.yuv444p_descriptor_sets[i],
                    dst_binding: 0,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &image_infos[i],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set: self.yuv444p_descriptor_sets[i],
                    dst_binding: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: 1,
                    p_buffer_info: &yuv444p_buffer_infos[i],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set: self.yuv444p_descriptor_sets[i],
                    dst_binding: 2,
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: 1,
                    p_buffer_info: &nv12_constants_buffer_info,
                    ..Default::default()
                },
            ]);
        }

        let mut video_nv12_input_infos = Vec::with_capacity(self.video_nv12_slots.len());
        let mut video_nv12_y_infos = Vec::with_capacity(self.video_nv12_slots.len());
        let mut video_nv12_uv_infos = Vec::with_capacity(self.video_nv12_slots.len());
        for slot in &self.video_nv12_slots {
            video_nv12_input_infos.push(general_info(self.render_targets[0].texture.view));
            video_nv12_y_infos.push(general_info(slot.image.y_view));
            video_nv12_uv_infos.push(general_info(slot.image.uv_view));
        }
        for i in 0..self.video_nv12_slots.len() {
            write_descriptor_sets.extend([
                vk::WriteDescriptorSet {
                    dst_set: self.video_nv12_slots[i].descriptor_set,
                    dst_binding: 0,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &video_nv12_input_infos[i],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set: self.video_nv12_slots[i].descriptor_set,
                    dst_binding: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &video_nv12_y_infos[i],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set: self.video_nv12_slots[i].descriptor_set,
                    dst_binding: 2,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: 1,
                    p_image_info: &video_nv12_uv_infos[i],
                    ..Default::default()
                },
            ]);
        }

        unsafe {
            resources
                .ctx
                .device
                .update_descriptor_sets(&write_descriptor_sets, &[]);
        }
    }

    pub(super) fn ensure_frame_attachments(&mut self, ctx: &Arc<VulkanContext>, msaa_samples: u32) {
        if self.requirements.depth && self.msaa_depth_texture.is_none() {
            self.msaa_depth_texture = Some(Image::new(
                ctx,
                self.requirements.raster_width,
                self.requirements.raster_height,
                vk::Format::D32_SFLOAT,
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                vk::ImageAspectFlags::DEPTH,
                msaa_to_vk_sample_count(msaa_samples),
            ));
            self.shared_image_states[GraphImageId::MsaaDepth.shared_index().unwrap()] =
                TrackedImageState::UNDEFINED;
        }

        let raster_sample_count = msaa_to_vk_sample_count(msaa_samples);
        if self.requirements.multisampled_raster
            && self.msaa_texture.is_none()
            && raster_sample_count != vk::SampleCountFlags::TYPE_1
        {
            self.msaa_texture = Some(Image::new(
                ctx,
                self.requirements.raster_width,
                self.requirements.raster_height,
                vk::Format::R16G16B16A16_SFLOAT,
                vk::ImageUsageFlags::COLOR_ATTACHMENT,
                vk::ImageAspectFlags::COLOR,
                raster_sample_count,
            ));
            self.shared_image_states[GraphImageId::MsaaColor.shared_index().unwrap()] =
                TrackedImageState::UNDEFINED;
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
        descriptor_sets.extend_from_slice(&self.nv12_descriptor_sets);
        descriptor_sets.extend_from_slice(&self.yuv444p_descriptor_sets);
        descriptor_sets.extend(self.video_nv12_slots.iter().map(|slot| slot.descriptor_set));
        self.descriptor_pool.free(&descriptor_sets);
    }
}
