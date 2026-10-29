use ash::vk;

use super::super::frame::write_gpu_timestamp;
use super::super::mesh_2d::{GeometryUpload2D, PreparedMesh2DBatch};
use super::super::output::RenderOutputs;
use super::super::prepared_frame::GpuGrid3D;
use super::super::targets::TargetCache;
use super::super::upload::UploadedFrame;
use super::super::{Image, Mesh3DDraw};
use super::graph::{
    BarrierTransition, ColorRasterGraphPass, GraphImageId, GraphPhase, ImageStateTable, PassKind,
    RenderGraph, RenderGraphPass,
};
use super::plan::RecordingPlan;
use super::{
    ColorAttachment, ColorRasterPass, CommandRecorder, DeferredOpaquePass, DepthAttachment,
    GeometryUploadBuffers2D, Grid3DBindings, Mesh2DBindings, Mesh3DBindings, TransparentDepthPass,
};

pub(in crate::vulkan::renderer) struct FrameRecord<'a> {
    pub(in crate::vulkan::renderer) plan: RecordingPlan,
    pub(in crate::vulkan::renderer) cache: &'a mut TargetCache,
    pub(in crate::vulkan::renderer) frame_index: usize,
    pub(in crate::vulkan::renderer) video_frame_index: usize,
    pub(in crate::vulkan::renderer) uploaded: UploadedFrame,
    pub(in crate::vulkan::renderer) outputs: RenderOutputs,
    pub(in crate::vulkan::renderer) mesh_draws_3d: &'a [Mesh3DDraw],
    pub(in crate::vulkan::renderer) grids_3d: &'a [GpuGrid3D],
    pub(in crate::vulkan::renderer) mesh_batches_2d: &'a [PreparedMesh2DBatch],
    pub(in crate::vulkan::renderer) geometry_uploads_2d: &'a [GeometryUpload2D],
    pub(in crate::vulkan::renderer) uploads_2d: GeometryUploadBuffers2D,
    pub(in crate::vulkan::renderer) mesh_3d_vertex: vk::Buffer,
    pub(in crate::vulkan::renderer) mesh_3d_index: vk::Buffer,
    pub(in crate::vulkan::renderer) mesh_2d_vertex: vk::Buffer,
    pub(in crate::vulkan::renderer) mesh_2d_index: vk::Buffer,
    pub(in crate::vulkan::renderer) mesh_2d_instance: vk::Buffer,
    pub(in crate::vulkan::renderer) raster_texture_set: vk::DescriptorSet,
    pub(in crate::vulkan::renderer) query_pool: vk::QueryPool,
}

struct FramePassContext<'a> {
    cache: &'a mut TargetCache,
    frame_index: usize,
    video_frame_index: usize,
    uploaded: &'a UploadedFrame,
    mesh_3d: Mesh3DBindings<'a>,
    mesh_2d: Mesh2DBindings<'a>,
    grids_3d: Grid3DBindings<'a>,
}

impl<'a> CommandRecorder<'a> {
    pub(in crate::vulkan::renderer) unsafe fn record_frame(&self, record: FrameRecord<'_>) {
        let FrameRecord {
            plan,
            cache,
            frame_index,
            video_frame_index,
            uploaded,
            outputs,
            mesh_draws_3d,
            grids_3d,
            mesh_batches_2d,
            geometry_uploads_2d,
            uploads_2d,
            mesh_3d_vertex,
            mesh_3d_index,
            mesh_2d_vertex,
            mesh_2d_index,
            mesh_2d_instance,
            raster_texture_set,
            query_pool,
        } = record;

        let raster_descriptor_set = cache.render_targets[frame_index].raster_descriptor_set;
        let grid_descriptor_set = cache.render_targets[frame_index].grid_descriptor_set;
        let mesh_3d = Mesh3DBindings {
            draws: mesh_draws_3d,
            descriptor_set: raster_descriptor_set,
            dynamic_offsets: &uploaded.raster_dynamic_offsets,
            vertex_buffer: mesh_3d_vertex,
            vertex_offset: uploaded.vertex_offset,
            index_buffer: mesh_3d_index,
            index_offset: uploaded.index_offset,
        };
        let mesh_2d = Mesh2DBindings {
            batches: mesh_batches_2d,
            camera_descriptor_set: cache.raster_descriptor_set_2d,
            camera_dynamic_offsets: &uploaded.raster_2d_dynamic_offsets,
            texture_descriptor_set: raster_texture_set,
            vertex_buffer: mesh_2d_vertex,
            index_buffer: mesh_2d_index,
            instance_buffer: mesh_2d_instance,
            instance_offset: uploaded.instance_2d_offset,
        };
        let grids_3d = Grid3DBindings {
            grids: grids_3d,
            descriptor_set: grid_descriptor_set,
            dynamic_offsets: &uploaded.grid_dynamic_offsets,
        };

        let mut state_table = ImageStateTable::from_parts(
            &cache.render_targets[frame_index].image_states,
            &cache.shared_image_states,
            outputs
                .vulkan_video
                .then(|| cache.video_nv12_slots[video_frame_index].layout),
        );
        let bloom_needs_clear = cache.render_targets[frame_index].bloom_contains_data
            || state_table.state(GraphImageId::BloomPing).layout == vk::ImageLayout::UNDEFINED;
        let graph = RenderGraph::compile(
            plan,
            outputs,
            cache.padded_bytes_per_row,
            bloom_needs_clear,
        );

        let mut ctx = FramePassContext {
            cache,
            frame_index,
            video_frame_index,
            uploaded: &uploaded,
            mesh_3d,
            mesh_2d,
            grids_3d,
        };

        unsafe {
            self.record_geometry_uploads_2d(geometry_uploads_2d, uploads_2d);
            write_gpu_timestamp(
                self.device,
                self.command_buffer,
                query_pool,
                1,
                plan.gpu_profiling,
            );

            for (phase, query) in [
                (GraphPhase::Sdf, Some(2)),
                (GraphPhase::SurfaceAndRaster, Some(3)),
                (GraphPhase::Postprocess, Some(4)),
                (GraphPhase::Output, Some(5)),
                (GraphPhase::Present, None),
            ] {
                for pass in graph.passes().iter().filter(|pass| pass.phase == phase) {
                    self.execute_graph_pass(pass, &mut ctx, &mut state_table);
                }
                if let Some(query) = query {
                    write_gpu_timestamp(
                        self.device,
                        self.command_buffer,
                        query_pool,
                        query,
                        plan.gpu_profiling,
                    );
                }
            }
        }

        state_table.write_back(
            &mut ctx.cache.render_targets[frame_index].image_states,
            &mut ctx.cache.shared_image_states,
            outputs
                .vulkan_video
                .then(|| &mut ctx.cache.video_nv12_slots[video_frame_index].layout),
        );
    }

    unsafe fn execute_graph_pass(
        &self,
        pass: &RenderGraphPass,
        ctx: &mut FramePassContext<'_>,
        state_table: &mut ImageStateTable,
    ) {
        let transitions = state_table.plan_pass_barriers(&pass.accesses, |id| {
            ctx.cache
                .vk_image(ctx.frame_index, ctx.video_frame_index, id)
                .is_some()
        });
        unsafe {
            self.emit_barriers(
                ctx.cache,
                ctx.frame_index,
                ctx.video_frame_index,
                &transitions,
            );

            match pass.kind {
                PassKind::ClearEmpty { background_color } => {
                    self.record_empty_frame(
                        &ctx.cache.render_targets[ctx.frame_index],
                        background_color,
                    );
                }
                PassKind::SdfRaymarch { extent } => {
                    self.record_sdf(
                        &ctx.cache.render_targets[ctx.frame_index],
                        &ctx.uploaded.compute_dynamic_offsets,
                        extent.width,
                        extent.height,
                    );
                }
                PassKind::DeferredOpaque {
                    region,
                    preserve_depth,
                } => {
                    let depth = ctx
                        .cache
                        .msaa_depth_texture
                        .as_ref()
                        .expect("deferred raster requires a depth attachment");
                    self.record_deferred_opaque(DeferredOpaquePass {
                        normal_depth: ctx.cache.raster_normal_depth.view,
                        albedo: ctx.cache.raster_albedo.view,
                        material_id: ctx.cache.raster_material_id.view,
                        depth: depth.view,
                        region,
                        preserve_depth,
                        meshes: ctx.mesh_3d,
                    });
                }
                PassKind::SurfaceResolve { extent } => {
                    self.record_surface_resolve(
                        &ctx.cache.render_targets[ctx.frame_index],
                        &ctx.uploaded.surface_dynamic_offsets,
                        extent,
                    );
                }
                PassKind::SurfaceLighting { extent } => {
                    self.record_surface_lighting(
                        &ctx.cache.render_targets[ctx.frame_index],
                        &ctx.uploaded.surface_dynamic_offsets,
                        extent,
                    );
                }
                PassKind::CopyColorImage { src, dst, extent } => {
                    let targets = &ctx.cache.render_targets[ctx.frame_index];
                    let src_image = targets.image(src).vk_image;
                    let dst_image = targets.image(dst).vk_image;
                    self.copy_color_image(src_image, dst_image, extent);
                }
                PassKind::TransparentBackDepth { extent } => {
                    let depth_view =
                        ctx.cache.render_targets[ctx.frame_index].transparent_back_depth.view;
                    self.record_transparent_depth(TransparentDepthPass {
                        depth: depth_view,
                        extent,
                        meshes: ctx.mesh_3d,
                    });
                }
                PassKind::ColorRaster(raster) => {
                    let ColorRasterGraphPass {
                        target,
                        use_msaa,
                        preserve_multisample,
                        color_load,
                        depth,
                        region,
                        meshes_3d,
                        grids_3d,
                        meshes_2d,
                    } = raster;
                    let target_view = ctx.cache.render_targets[ctx.frame_index].image(target).view;
                    let color = Self::color_attachment(
                        use_msaa
                            .then_some(ctx.cache.msaa_texture.as_ref())
                            .flatten(),
                        target_view,
                        preserve_multisample,
                    );
                    let depth = depth.map(|(load, preserve)| {
                        let depth_tex = ctx
                            .cache
                            .msaa_depth_texture
                            .as_ref()
                            .expect("3D/overlay raster requires a depth attachment");
                        DepthAttachment {
                            view: depth_tex.view,
                            load,
                            preserve,
                        }
                    });
                    self.record_color_raster(ColorRasterPass {
                        color,
                        color_load,
                        depth,
                        region,
                        meshes_3d: meshes_3d.map(|pass| (pass, ctx.mesh_3d)),
                        grids_3d: grids_3d.then_some(ctx.grids_3d),
                        meshes_2d: meshes_2d.map(|pass| (pass, ctx.mesh_2d)),
                    });
                }
                PassKind::SurfaceComposite {
                    has_overlay,
                    width,
                    height,
                    raster_scale,
                } => {
                    self.record_surface_composite(
                        &ctx.cache.render_targets[ctx.frame_index],
                        has_overlay,
                        width,
                        height,
                        raster_scale,
                    );
                }
                PassKind::BloomExtract => {
                    self.record_bloom_extract(&ctx.cache.render_targets[ctx.frame_index]);
                }
                PassKind::BloomHorizontal => {
                    self.record_bloom_horizontal(&ctx.cache.render_targets[ctx.frame_index]);
                }
                PassKind::BloomVertical => {
                    self.record_bloom_vertical(&mut ctx.cache.render_targets[ctx.frame_index]);
                }
                PassKind::BloomClear => {
                    self.record_bloom_clear(&mut ctx.cache.render_targets[ctx.frame_index]);
                }
                PassKind::ToneMap { width, height } => {
                    self.record_tone_map(&ctx.cache.render_targets[ctx.frame_index], width, height);
                }
                PassKind::OutputCpuNv12 { width, height } => {
                    let set = ctx.cache.nv12_descriptor_sets[ctx.frame_index];
                    self.record_output_cpu_nv12(set, width, height);
                }
                PassKind::OutputCpuYuv444p { width, height } => {
                    let set = ctx.cache.yuv444p_descriptor_sets[ctx.frame_index];
                    self.record_output_cpu_yuv444p(set, width, height);
                }
                PassKind::OutputVulkanVideo {
                    width,
                    height,
                    fused_downsample,
                } => {
                    let set = ctx.cache.video_nv12_slots[ctx.video_frame_index].descriptor_set;
                    self.record_output_vulkan_video(set, width, height, fused_downsample);
                }
                PassKind::CopyRgbaToBuffer {
                    width,
                    height,
                    padded_bytes_per_row,
                } => {
                    let src_image = ctx.cache.render_targets[ctx.frame_index].texture.vk_image;
                    let dst_buffer = ctx.cache.output_buffers[ctx.frame_index].vk_buffer;
                    self.record_copy_rgba_to_buffer(
                        src_image,
                        dst_buffer,
                        width,
                        height,
                        padded_bytes_per_row,
                    );
                }
                PassKind::PresentReady => {}
            }
        }
    }

    unsafe fn emit_barriers(
        &self,
        cache: &TargetCache,
        frame_index: usize,
        video_frame_index: usize,
        transitions: &[BarrierTransition],
    ) {
        if transitions.is_empty() {
            return;
        }
        let mut barriers = Vec::with_capacity(transitions.len());
        for transition in transitions {
            let Some(image) = cache.vk_image(frame_index, video_frame_index, transition.image)
            else {
                continue;
            };
            barriers.push(
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(transition.old.stage)
                    .src_access_mask(transition.old.access)
                    .dst_stage_mask(transition.new.stage)
                    .dst_access_mask(transition.new.access)
                    .old_layout(transition.old.layout)
                    .new_layout(transition.new.layout)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(image)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: transition.aspect_mask,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    }),
            );
        }
        if barriers.is_empty() {
            return;
        }
        let dependency = vk::DependencyInfo::default().image_memory_barriers(&barriers);
        unsafe {
            self.device
                .cmd_pipeline_barrier2(self.command_buffer, &dependency);
        }
    }

    fn color_attachment(
        msaa: Option<&Image>,
        resolved_view: vk::ImageView,
        preserve_multisample: bool,
    ) -> ColorAttachment {
        match msaa {
            Some(msaa_texture) => ColorAttachment::Resolve {
                multisample: msaa_texture.view,
                resolved: resolved_view,
                preserve_multisample,
            },
            None => ColorAttachment::Single(resolved_view),
        }
    }

    unsafe fn copy_color_image(&self, src: vk::Image, dst: vk::Image, extent: vk::Extent2D) {
        unsafe {
            self.device.cmd_copy_image(
                self.command_buffer,
                src,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst,
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
                        width: extent.width,
                        height: extent.height,
                        depth: 1,
                    })],
            );
        }
    }
}
