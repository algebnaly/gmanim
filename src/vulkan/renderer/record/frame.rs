use ash::vk;

use super::super::frame::{
    FrameExecutionPlan, TrackedImageState, transition_image, write_gpu_timestamp,
};
use super::super::mesh_2d::{GeometryUpload2D, PreparedMesh2DBatch};
use super::super::output::RenderOutputs;
use super::super::prepared_frame::GpuGrid3D;
use super::super::targets::TargetCache;
use super::super::upload::UploadedFrame;
use super::super::{Image, Mesh3DDraw};
use super::plan::RecordingPlan;
use super::{
    ColorAttachment, ColorLoad, ColorRasterPass, CommandRecorder, DeferredOpaquePass,
    DepthAttachment, DepthLoad, GeometryUploadBuffers2D, Grid3DBindings, Mesh2DBindings,
    Mesh2DPass, Mesh3DBindings, Mesh3DPass, OutputPasses, RasterAttachment, RasterRegion,
    TransparentDepthPass, VideoOutputPass,
};

const COLOR_ATTACHMENT_STATE: TrackedImageState = TrackedImageState {
    layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
    access: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
};
const COMPUTE_READ_STATE: TrackedImageState = TrackedImageState {
    layout: vk::ImageLayout::GENERAL,
    stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
    access: vk::AccessFlags2::SHADER_READ,
};
const COPY_SRC_STATE: TrackedImageState = TrackedImageState {
    layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
    stage: vk::PipelineStageFlags2::COPY,
    access: vk::AccessFlags2::TRANSFER_READ,
};
const COPY_DST_STATE: TrackedImageState = TrackedImageState {
    layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
    stage: vk::PipelineStageFlags2::COPY,
    access: vk::AccessFlags2::TRANSFER_WRITE,
};
const FRAGMENT_SHADER_READ_STATE: TrackedImageState = TrackedImageState {
    layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
    access: vk::AccessFlags2::SHADER_READ,
};
const SDF_DEPTH_FRAGMENT_READ_STATE: TrackedImageState = TrackedImageState {
    layout: vk::ImageLayout::GENERAL,
    stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
    access: vk::AccessFlags2::SHADER_READ,
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
        let targets = &mut cache.render_targets[frame_index];
        let mesh_3d = Mesh3DBindings {
            draws: mesh_draws_3d,
            descriptor_set: targets.raster_descriptor_set,
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
            raster_scale: plan.raster_scale,
            descriptor_set: targets.grid_descriptor_set,
            dynamic_offsets: &uploaded.grid_dynamic_offsets,
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

            if plan.execution == FrameExecutionPlan::Empty {
                self.record_empty_frame(targets, plan.background_color);
            }

            if plan.execution.runs_sdf() {
                let sdf_extent = plan.ssaa_extent();
                self.record_sdf(
                    targets,
                    &uploaded.compute_dynamic_offsets,
                    sdf_extent.width,
                    sdf_extent.height,
                );
            }
            write_gpu_timestamp(
                self.device,
                self.command_buffer,
                query_pool,
                2,
                plan.gpu_profiling,
            );

            if plan.uses_deferred_raster {
                let raster_extent = plan.ssaa_extent();
                let (depth_image, depth_view) = {
                    let depth = cache
                        .msaa_depth_texture
                        .as_ref()
                        .expect("deferred raster requires a depth attachment");
                    (depth.vk_image, depth.view)
                };
                self.record_deferred_opaque(DeferredOpaquePass {
                    normal_depth: RasterAttachment {
                        image: cache.raster_normal_depth.vk_image,
                        view: cache.raster_normal_depth.view,
                        state: &mut cache.raster_normal_depth_state,
                    },
                    albedo: RasterAttachment {
                        image: cache.raster_albedo.vk_image,
                        view: cache.raster_albedo.view,
                        state: &mut cache.raster_albedo_state,
                    },
                    material_id: RasterAttachment {
                        image: cache.raster_material_id.vk_image,
                        view: cache.raster_material_id.view,
                        state: &mut cache.raster_material_id_state,
                    },
                    depth: RasterAttachment {
                        image: depth_image,
                        view: depth_view,
                        state: &mut cache.msaa_depth_texture_state,
                    },
                    region: RasterRegion::new(
                        raster_extent,
                        plan.camera_clip,
                        plan.camera_raster_scale,
                    ),
                    preserve_depth: plan.has_transparent_meshes || plan.has_grid_3d,
                    meshes: mesh_3d,
                });
                if !plan.execution.runs_sdf() {
                    self.transition_sdf_gbuffer(targets, COMPUTE_READ_STATE);
                }
                self.record_surface_compute(
                    targets,
                    &uploaded.surface_dynamic_offsets,
                    raster_extent,
                );

                if plan.has_surface_overlay {
                    if plan.execution.runs_sdf()
                        && (plan.has_transparent_meshes || plan.has_grid_3d)
                    {
                        transition_image(
                            self.device,
                            self.command_buffer,
                            targets.sdf_depth.vk_image,
                            vk::ImageAspectFlags::COLOR,
                            &mut targets.sdf_depth_state,
                            SDF_DEPTH_FRAGMENT_READ_STATE,
                        );
                    }
                    if plan.has_transparent_meshes {
                        self.copy_color_image(
                            targets.surface_hdr.vk_image,
                            &mut targets.surface_hdr_state,
                            targets.scene_color.vk_image,
                            &mut targets.scene_color_state,
                            raster_extent,
                            None,
                        );
                        self.record_transparent_depth(TransparentDepthPass {
                            depth: RasterAttachment {
                                image: targets.transparent_back_depth.vk_image,
                                view: targets.transparent_back_depth.view,
                                state: &mut targets.transparent_back_depth_state,
                            },
                            extent: raster_extent,
                            meshes: mesh_3d,
                        });
                    }

                    let overlay_color = Self::color_attachment(
                        cache.msaa_texture.as_ref(),
                        &mut cache.msaa_texture_state,
                        &targets.overlay_hdr,
                        &mut targets.overlay_hdr_state,
                        false,
                    );
                    let overlay_depth = cache
                        .msaa_depth_texture
                        .as_ref()
                        .expect("deferred overlay requires a depth attachment");
                    self.record_color_raster(ColorRasterPass {
                        color: overlay_color,
                        color_load: ColorLoad::Clear([0.0; 4]),
                        depth: Some(DepthAttachment {
                            attachment: RasterAttachment {
                                image: overlay_depth.vk_image,
                                view: overlay_depth.view,
                                state: &mut cache.msaa_depth_texture_state,
                            },
                            load: if plan.has_transparent_meshes || plan.has_grid_3d {
                                DepthLoad::Load
                            } else {
                                DepthLoad::Discard
                            },
                            preserve: false,
                        }),
                        region: RasterRegion::new(raster_extent, None, 1.0),
                        meshes_3d: plan
                            .has_transparent_meshes
                            .then_some((Mesh3DPass::TransparentColor, mesh_3d)),
                        grids_3d: plan.has_grid_3d.then_some(grids_3d),
                        meshes_2d: Some((Mesh2DPass::Depth, mesh_2d)),
                    });
                }
            }

            if plan.execution.runs_sdf() && !plan.uses_deferred_raster {
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
                        self.device,
                        self.command_buffer,
                        image,
                        vk::ImageAspectFlags::COLOR,
                        state,
                        COMPUTE_READ_STATE,
                    );
                }
                self.record_surface_compute(
                    targets,
                    &uploaded.surface_dynamic_offsets,
                    plan.ssaa_extent(),
                );
                if plan.has_transparent_meshes || plan.has_grid_3d {
                    transition_image(
                        self.device,
                        self.command_buffer,
                        targets.sdf_depth.vk_image,
                        vk::ImageAspectFlags::COLOR,
                        &mut targets.sdf_depth_state,
                        SDF_DEPTH_FRAGMENT_READ_STATE,
                    );
                }
            }

            if plan.execution.runs_raster() && !plan.uses_deferred_raster {
                let raster_extent = plan.raster_extent();
                let overlay_destination = plan.execution == FrameExecutionPlan::SdfRasterComposite;
                let (target_image, target_view, target_state) = if overlay_destination {
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
                let color = Self::color_attachment_views(
                    cache.msaa_texture.as_ref(),
                    &mut cache.msaa_texture_state,
                    target_image,
                    target_view,
                    target_state,
                    true,
                );
                let depth = plan.raster_uses_depth.then(|| {
                    let texture = cache
                        .msaa_depth_texture
                        .as_ref()
                        .expect("3D raster requires a depth attachment");
                    DepthAttachment {
                        attachment: RasterAttachment {
                            image: texture.vk_image,
                            view: texture.view,
                            state: &mut cache.msaa_depth_texture_state,
                        },
                        load: DepthLoad::Clear,
                        preserve: plan.has_transparent_meshes,
                    }
                });
                self.record_color_raster(ColorRasterPass {
                    color,
                    color_load: ColorLoad::Clear(plan.background_color),
                    depth,
                    region: RasterRegion::new(
                        raster_extent,
                        plan.camera_clip,
                        plan.camera_raster_scale,
                    ),
                    meshes_3d: None,
                    grids_3d: plan.has_grid_3d.then_some(grids_3d),
                    meshes_2d: (!plan.has_transparent_meshes).then_some((
                        if plan.analytic_2d {
                            Mesh2DPass::Analytic
                        } else if plan.raster_uses_depth {
                            Mesh2DPass::Depth
                        } else {
                            Mesh2DPass::Depthless
                        },
                        mesh_2d,
                    )),
                });

                if plan.has_transparent_meshes {
                    let restore_src = if plan.execution.runs_sdf() {
                        COMPUTE_READ_STATE
                    } else {
                        COLOR_ATTACHMENT_STATE
                    };
                    if plan.execution.runs_sdf() {
                        self.copy_color_image(
                            targets.surface_hdr.vk_image,
                            &mut targets.surface_hdr_state,
                            targets.scene_color.vk_image,
                            &mut targets.scene_color_state,
                            raster_extent,
                            Some(restore_src),
                        );
                    } else if overlay_destination {
                        self.copy_color_image(
                            targets.overlay_hdr.vk_image,
                            &mut targets.overlay_hdr_state,
                            targets.scene_color.vk_image,
                            &mut targets.scene_color_state,
                            raster_extent,
                            Some(restore_src),
                        );
                    } else {
                        self.copy_color_image(
                            targets.resolved_texture.vk_image,
                            &mut targets.resolved_texture_state,
                            targets.scene_color.vk_image,
                            &mut targets.scene_color_state,
                            raster_extent,
                            Some(restore_src),
                        );
                    }

                    self.record_transparent_depth(TransparentDepthPass {
                        depth: RasterAttachment {
                            image: targets.transparent_back_depth.vk_image,
                            view: targets.transparent_back_depth.view,
                            state: &mut targets.transparent_back_depth_state,
                        },
                        extent: raster_extent,
                        meshes: mesh_3d,
                    });

                    let (target_image, target_view, target_state) = if overlay_destination {
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
                    let color = Self::color_attachment_views(
                        cache.msaa_texture.as_ref(),
                        &mut cache.msaa_texture_state,
                        target_image,
                        target_view,
                        target_state,
                        false,
                    );
                    let depth = plan.raster_uses_depth.then(|| {
                        let texture = cache
                            .msaa_depth_texture
                            .as_ref()
                            .expect("transparent 3D raster requires a depth attachment");
                        DepthAttachment {
                            attachment: RasterAttachment {
                                image: texture.vk_image,
                                view: texture.view,
                                state: &mut cache.msaa_depth_texture_state,
                            },
                            load: DepthLoad::Load,
                            preserve: false,
                        }
                    });
                    self.record_color_raster(ColorRasterPass {
                        color,
                        color_load: ColorLoad::Load,
                        depth,
                        region: RasterRegion::new(
                            raster_extent,
                            plan.camera_clip,
                            plan.camera_raster_scale,
                        ),
                        meshes_3d: Some((Mesh3DPass::TransparentColor, mesh_3d)),
                        grids_3d: None,
                        meshes_2d: Some((Mesh2DPass::Depth, mesh_2d)),
                    });
                }
            }

            if plan.execution.runs_sdf() || plan.uses_deferred_raster {
                self.record_surface_composite(
                    targets,
                    plan.has_surface_overlay,
                    plan.width,
                    plan.height,
                    plan.ssaa_factor,
                );
            }
            write_gpu_timestamp(
                self.device,
                self.command_buffer,
                query_pool,
                3,
                plan.gpu_profiling,
            );

            if plan.execution != FrameExecutionPlan::Empty {
                self.record_bloom(targets, plan.bloom_enabled);
            }

            if plan.runs_postprocess {
                self.record_tone_map(targets, plan.width, plan.height);
            }
            write_gpu_timestamp(
                self.device,
                self.command_buffer,
                query_pool,
                4,
                plan.gpu_profiling,
            );

            let has_compute_output =
                outputs.cpu_nv12 || outputs.cpu_yuv444p || outputs.vulkan_video;
            let video_output = outputs.vulkan_video.then(|| {
                let slot = &cache.video_nv12_slots[video_frame_index];
                VideoOutputPass {
                    image: slot.image.vk_image,
                    descriptor_set: slot.descriptor_set,
                    current_layout: slot.layout,
                }
            });
            self.record_outputs(
                targets,
                OutputPasses {
                    width: plan.width,
                    height: plan.height,
                    fused_video_downsample: plan.fused_video_downsample,
                    cpu_nv12_descriptor_set: outputs
                        .cpu_nv12
                        .then_some(cache.nv12_descriptor_sets[frame_index]),
                    cpu_yuv444p_descriptor_set: outputs
                        .cpu_yuv444p
                        .then_some(cache.yuv444p_descriptor_sets[frame_index]),
                    video: video_output,
                    rgba_buffer: outputs
                        .cpu_rgba
                        .then_some(cache.output_buffers[frame_index].vk_buffer),
                    rgba_padded_bytes_per_row: cache.padded_bytes_per_row,
                },
            );
            if outputs.vulkan_video {
                cache.video_nv12_slots[video_frame_index].layout = vk::ImageLayout::GENERAL;
            }
            write_gpu_timestamp(
                self.device,
                self.command_buffer,
                query_pool,
                5,
                plan.gpu_profiling,
            );

            if !has_compute_output && !outputs.cpu_rgba {
                transition_image(
                    self.device,
                    self.command_buffer,
                    targets.texture.vk_image,
                    vk::ImageAspectFlags::COLOR,
                    &mut targets.texture_state,
                    FRAGMENT_SHADER_READ_STATE,
                );
            }
        }
    }

    fn color_attachment<'s>(
        msaa: Option<&'s Image>,
        msaa_state: &'s mut TrackedImageState,
        resolved: &'s Image,
        resolved_state: &'s mut TrackedImageState,
        preserve_multisample: bool,
    ) -> ColorAttachment<'s> {
        Self::color_attachment_views(
            msaa,
            msaa_state,
            resolved.vk_image,
            resolved.view,
            resolved_state,
            preserve_multisample,
        )
    }

    fn color_attachment_views<'s>(
        msaa: Option<&'s Image>,
        msaa_state: &'s mut TrackedImageState,
        resolved_image: vk::Image,
        resolved_view: vk::ImageView,
        resolved_state: &'s mut TrackedImageState,
        preserve_multisample: bool,
    ) -> ColorAttachment<'s> {
        match msaa {
            Some(msaa_texture) => ColorAttachment::Resolve {
                multisample: RasterAttachment {
                    image: msaa_texture.vk_image,
                    view: msaa_texture.view,
                    state: msaa_state,
                },
                resolved: RasterAttachment {
                    image: resolved_image,
                    view: resolved_view,
                    state: resolved_state,
                },
                preserve_multisample,
            },
            None => ColorAttachment::Single(RasterAttachment {
                image: resolved_image,
                view: resolved_view,
                state: resolved_state,
            }),
        }
    }

    unsafe fn transition_sdf_gbuffer(
        &self,
        targets: &mut super::super::targets::RenderTargetSet,
        destination: TrackedImageState,
    ) {
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
            unsafe {
                transition_image(
                    self.device,
                    self.command_buffer,
                    image,
                    vk::ImageAspectFlags::COLOR,
                    state,
                    destination,
                );
            }
        }
    }

    unsafe fn copy_color_image(
        &self,
        src: vk::Image,
        src_state: &mut TrackedImageState,
        dst: vk::Image,
        dst_state: &mut TrackedImageState,
        extent: vk::Extent2D,
        restore_src: Option<TrackedImageState>,
    ) {
        unsafe {
            transition_image(
                self.device,
                self.command_buffer,
                src,
                vk::ImageAspectFlags::COLOR,
                src_state,
                COPY_SRC_STATE,
            );
            transition_image(
                self.device,
                self.command_buffer,
                dst,
                vk::ImageAspectFlags::COLOR,
                dst_state,
                COPY_DST_STATE,
            );
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
            if let Some(restore_src) = restore_src {
                transition_image(
                    self.device,
                    self.command_buffer,
                    src,
                    vk::ImageAspectFlags::COLOR,
                    src_state,
                    restore_src,
                );
            }
            transition_image(
                self.device,
                self.command_buffer,
                dst,
                vk::ImageAspectFlags::COLOR,
                dst_state,
                FRAGMENT_SHADER_READ_STATE,
            );
        }
    }
}
