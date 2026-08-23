use ash::vk;

use super::super::frame::{TrackedImageState, transition_image};
use super::{
    ColorAttachment, ColorLoad, ColorRasterPass, CommandRecorder, DeferredOpaquePass,
    DepthAttachment, DepthLoad, Mesh2DBindings, Mesh2DPass, Mesh3DBindings, Mesh3DPass,
    RasterAttachment, TransparentDepthPass,
};

const COLOR_ATTACHMENT_STATE: TrackedImageState = TrackedImageState {
    layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
    access: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
};

fn depth_attachment_state() -> TrackedImageState {
    TrackedImageState {
        layout: vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
        stage: vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
            | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
        access: vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
            | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
    }
}

impl<'a> CommandRecorder<'a> {
    pub(in crate::vulkan::renderer) unsafe fn record_color_raster(
        &self,
        pass: ColorRasterPass<'_>,
    ) {
        let ColorRasterPass {
            color,
            color_load,
            depth,
            region,
            meshes_3d,
            meshes_2d,
        } = pass;

        let color_info = unsafe {
            match color {
                ColorAttachment::Single(mut attachment) => {
                    self.transition_color_attachment(&mut attachment);
                    Self::color_attachment_info(attachment.view, color_load)
                        .store_op(vk::AttachmentStoreOp::STORE)
                }
                ColorAttachment::Resolve {
                    mut multisample,
                    mut resolved,
                    preserve_multisample,
                } => {
                    self.transition_color_attachment(&mut multisample);
                    self.transition_color_attachment(&mut resolved);
                    Self::color_attachment_info(multisample.view, color_load)
                        .store_op(if preserve_multisample {
                            vk::AttachmentStoreOp::STORE
                        } else {
                            vk::AttachmentStoreOp::DONT_CARE
                        })
                        .resolve_mode(vk::ResolveModeFlags::AVERAGE)
                        .resolve_image_view(resolved.view)
                        .resolve_image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                }
            }
        };

        let depth_info = unsafe {
            match depth {
                Some(DepthAttachment {
                    mut attachment,
                    load,
                    preserve,
                }) => {
                    transition_image(
                        self.device,
                        self.command_buffer,
                        attachment.image,
                        vk::ImageAspectFlags::DEPTH,
                        attachment.state,
                        depth_attachment_state(),
                    );
                    let load_op = match load {
                        DepthLoad::Clear => vk::AttachmentLoadOp::CLEAR,
                        DepthLoad::Load => vk::AttachmentLoadOp::LOAD,
                        DepthLoad::Discard => vk::AttachmentLoadOp::DONT_CARE,
                    };
                    Some(
                        vk::RenderingAttachmentInfo::default()
                            .image_view(attachment.view)
                            .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                            .load_op(load_op)
                            .store_op(if preserve {
                                vk::AttachmentStoreOp::STORE
                            } else {
                                vk::AttachmentStoreOp::DONT_CARE
                            })
                            .clear_value(vk::ClearValue {
                                depth_stencil: vk::ClearDepthStencilValue {
                                    depth: 1.0,
                                    stencil: 0,
                                },
                            }),
                    )
                }
                None => None,
            }
        };

        let color_infos = [color_info];
        let mut rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: region.extent,
            })
            .layer_count(1)
            .color_attachments(&color_infos);
        if let Some(depth_info) = depth_info.as_ref() {
            rendering_info = rendering_info.depth_attachment(depth_info);
        }

        unsafe {
            self.device
                .cmd_begin_rendering(self.command_buffer, &rendering_info);
            self.device.cmd_set_viewport(
                self.command_buffer,
                0,
                std::slice::from_ref(&region.viewport),
            );
            self.device.cmd_set_scissor(
                self.command_buffer,
                0,
                std::slice::from_ref(&region.scissor),
            );
            if let Some((mesh_pass, bindings)) = meshes_3d {
                self.record_meshes_3d(mesh_pass, bindings);
            }
            if let Some((mesh_pass, bindings)) = meshes_2d {
                self.record_meshes_2d(mesh_pass, bindings);
            }
            self.device.cmd_end_rendering(self.command_buffer);
        }
    }

    unsafe fn transition_color_attachment(&self, attachment: &mut RasterAttachment<'_>) {
        unsafe {
            transition_image(
                self.device,
                self.command_buffer,
                attachment.image,
                vk::ImageAspectFlags::COLOR,
                attachment.state,
                COLOR_ATTACHMENT_STATE,
            );
        }
    }

    fn color_attachment_info(
        view: vk::ImageView,
        load: ColorLoad,
    ) -> vk::RenderingAttachmentInfo<'static> {
        let info = vk::RenderingAttachmentInfo::default()
            .image_view(view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        match load {
            ColorLoad::Clear(color) => {
                info.load_op(vk::AttachmentLoadOp::CLEAR)
                    .clear_value(vk::ClearValue {
                        color: vk::ClearColorValue { float32: color },
                    })
            }
            ColorLoad::Load => info.load_op(vk::AttachmentLoadOp::LOAD),
        }
    }

    pub(in crate::vulkan::renderer) unsafe fn record_meshes_3d(
        &self,
        pass: Mesh3DPass,
        bindings: Mesh3DBindings<'_>,
    ) {
        unsafe {
            self.device.cmd_bind_descriptor_sets(
                self.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipelines.raster_pipeline_layout,
                0,
                std::slice::from_ref(&bindings.descriptor_set),
                bindings.dynamic_offsets,
            );
            self.device.cmd_bind_vertex_buffers(
                self.command_buffer,
                0,
                std::slice::from_ref(&bindings.vertex_buffer),
                &[bindings.vertex_offset],
            );
            self.device.cmd_bind_index_buffer(
                self.command_buffer,
                bindings.index_buffer,
                bindings.index_offset,
                vk::IndexType::UINT32,
            );

            match pass {
                Mesh3DPass::Opaque => {
                    self.device.cmd_bind_pipeline(
                        self.command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipelines.raster_pipeline,
                    );
                    for draw in bindings.draws.iter().filter(|draw| !draw.is_transparent()) {
                        self.record_mesh_3d_draw(draw);
                    }
                }
                Mesh3DPass::TransparentDepth => {
                    self.device.cmd_bind_pipeline(
                        self.command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipelines.raster_pipeline_transparent_depth,
                    );
                    for draw in bindings.draws.iter().filter(|draw| draw.is_transparent()) {
                        self.record_mesh_3d_draw(draw);
                    }
                }
                Mesh3DPass::TransparentColor => {
                    for draw in bindings.draws.iter().filter(|draw| draw.is_transparent()) {
                        for pipeline in [
                            self.pipelines.raster_pipeline_transparent_back,
                            self.pipelines.raster_pipeline_transparent_front,
                        ] {
                            self.device.cmd_bind_pipeline(
                                self.command_buffer,
                                vk::PipelineBindPoint::GRAPHICS,
                                pipeline,
                            );
                            self.record_mesh_3d_draw(draw);
                        }
                    }
                }
            }
        }
    }

    pub(in crate::vulkan::renderer) unsafe fn record_meshes_2d(
        &self,
        pass: Mesh2DPass,
        bindings: Mesh2DBindings<'_>,
    ) {
        if bindings.batches.is_empty() {
            return;
        }
        let pipeline = match pass {
            Mesh2DPass::Depth => self.pipelines.raster_pipeline_2d,
            Mesh2DPass::Depthless => self.pipelines.raster_pipeline_2d_depthless,
            Mesh2DPass::Analytic => self.pipelines.raster_pipeline_2d_analytic,
        };
        unsafe {
            self.device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                self.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipelines.raster_pipeline_layout_2d,
                0,
                std::slice::from_ref(&bindings.camera_descriptor_set),
                bindings.camera_dynamic_offsets,
            );
            self.device.cmd_bind_descriptor_sets(
                self.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipelines.raster_pipeline_layout_2d,
                1,
                std::slice::from_ref(&bindings.texture_descriptor_set),
                &[],
            );
            self.device.cmd_bind_vertex_buffers(
                self.command_buffer,
                0,
                &[bindings.vertex_buffer, bindings.instance_buffer],
                &[0, bindings.instance_offset],
            );
            self.device.cmd_bind_index_buffer(
                self.command_buffer,
                bindings.index_buffer,
                0,
                vk::IndexType::UINT32,
            );
            for batch in bindings.batches {
                self.device.cmd_draw_indexed(
                    self.command_buffer,
                    batch.index_count,
                    batch.instance_count,
                    batch.first_index,
                    batch.vertex_offset,
                    batch.first_instance,
                );
            }
        }
    }

    pub(in crate::vulkan::renderer) unsafe fn record_deferred_opaque(
        &self,
        pass: DeferredOpaquePass<'_>,
    ) {
        let DeferredOpaquePass {
            mut normal_depth,
            mut albedo,
            mut material_id,
            depth,
            region,
            preserve_depth,
            meshes,
        } = pass;
        let color_attachment = TrackedImageState {
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            access: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        };
        let compute_read = TrackedImageState {
            layout: vk::ImageLayout::GENERAL,
            stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
            access: vk::AccessFlags2::SHADER_READ,
        };
        let depth_attachment = TrackedImageState {
            layout: vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
            stage: vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            access: vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
                | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
        };

        unsafe {
            for attachment in [&mut normal_depth, &mut albedo, &mut material_id] {
                transition_image(
                    self.device,
                    self.command_buffer,
                    attachment.image,
                    vk::ImageAspectFlags::COLOR,
                    attachment.state,
                    color_attachment,
                );
            }
            transition_image(
                self.device,
                self.command_buffer,
                depth.image,
                vk::ImageAspectFlags::DEPTH,
                depth.state,
                depth_attachment,
            );

            let color_attachments = [
                vk::RenderingAttachmentInfo::default()
                    .image_view(normal_depth.view)
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(vk::ClearValue {
                        color: vk::ClearColorValue { float32: [0.0; 4] },
                    }),
                vk::RenderingAttachmentInfo::default()
                    .image_view(albedo.view)
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(vk::ClearValue {
                        color: vk::ClearColorValue { float32: [0.0; 4] },
                    }),
                vk::RenderingAttachmentInfo::default()
                    .image_view(material_id.view)
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(vk::ClearValue {
                        color: vk::ClearColorValue { uint32: [0; 4] },
                    }),
            ];
            let depth_info = vk::RenderingAttachmentInfo::default()
                .image_view(depth.view)
                .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(if preserve_depth {
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
            self.device.cmd_begin_rendering(
                self.command_buffer,
                &vk::RenderingInfo::default()
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: region.extent,
                    })
                    .layer_count(1)
                    .color_attachments(&color_attachments)
                    .depth_attachment(&depth_info),
            );
            self.device.cmd_set_viewport(
                self.command_buffer,
                0,
                std::slice::from_ref(&region.viewport),
            );
            self.device.cmd_set_scissor(
                self.command_buffer,
                0,
                std::slice::from_ref(&region.scissor),
            );
            self.record_meshes_3d(Mesh3DPass::Opaque, meshes);
            self.device.cmd_end_rendering(self.command_buffer);

            for attachment in [&mut normal_depth, &mut albedo, &mut material_id] {
                transition_image(
                    self.device,
                    self.command_buffer,
                    attachment.image,
                    vk::ImageAspectFlags::COLOR,
                    attachment.state,
                    compute_read,
                );
            }
        }
    }

    pub(in crate::vulkan::renderer) unsafe fn record_transparent_depth(
        &self,
        pass: TransparentDepthPass<'_>,
    ) {
        let color_attachment = TrackedImageState {
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            access: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        };
        let fragment_read = TrackedImageState {
            layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
            access: vk::AccessFlags2::SHADER_READ,
        };
        unsafe {
            transition_image(
                self.device,
                self.command_buffer,
                pass.depth.image,
                vk::ImageAspectFlags::COLOR,
                pass.depth.state,
                color_attachment,
            );
            let attachment = vk::RenderingAttachmentInfo::default()
                .image_view(pass.depth.view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue {
                    color: vk::ClearColorValue { float32: [0.0; 4] },
                });
            self.device.cmd_begin_rendering(
                self.command_buffer,
                &vk::RenderingInfo::default()
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: pass.extent,
                    })
                    .layer_count(1)
                    .color_attachments(std::slice::from_ref(&attachment)),
            );
            self.record_meshes_3d(Mesh3DPass::TransparentDepth, pass.meshes);
            self.device.cmd_end_rendering(self.command_buffer);
            transition_image(
                self.device,
                self.command_buffer,
                pass.depth.image,
                vk::ImageAspectFlags::COLOR,
                pass.depth.state,
                fragment_read,
            );
        }
    }
}
