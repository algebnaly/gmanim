use ash::vk;

use super::Mesh3DDraw;
use super::frame::{TrackedImageState, transition_image};
use super::mesh_2d::{GeometryUpload2D, PreparedMesh2DBatch};
use super::pipelines::PipelineSet;
use super::targets::RenderTargetSet;

pub(super) struct CommandRecorder<'a> {
    device: &'a ash::Device,
    command_buffer: vk::CommandBuffer,
    pipelines: &'a PipelineSet,
}

#[derive(Clone, Copy)]
pub(super) struct VideoOutputPass {
    pub(super) image: vk::Image,
    pub(super) descriptor_set: vk::DescriptorSet,
    pub(super) current_layout: vk::ImageLayout,
}

#[derive(Clone, Copy)]
pub(super) struct OutputPasses {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) fused_video_downsample: bool,
    pub(super) cpu_nv12_descriptor_set: Option<vk::DescriptorSet>,
    pub(super) cpu_yuv444p_descriptor_set: Option<vk::DescriptorSet>,
    pub(super) video: Option<VideoOutputPass>,
    pub(super) rgba_buffer: Option<vk::Buffer>,
    pub(super) rgba_padded_bytes_per_row: u32,
}

#[derive(Clone, Copy)]
pub(super) struct GeometryUploadBuffers2D {
    pub(super) vertex_staging: vk::Buffer,
    pub(super) vertex_staging_base: u64,
    pub(super) index_staging: vk::Buffer,
    pub(super) index_staging_base: u64,
    pub(super) vertex_device: vk::Buffer,
    pub(super) index_device: vk::Buffer,
}

#[derive(Clone, Copy)]
pub(super) enum Mesh3DPass {
    Opaque,
    TransparentDepth,
    TransparentColor,
}

pub(super) struct Mesh3DBindings<'a> {
    pub(super) draws: &'a [Mesh3DDraw],
    pub(super) descriptor_set: vk::DescriptorSet,
    pub(super) dynamic_offsets: &'a [u32],
    pub(super) vertex_buffer: vk::Buffer,
    pub(super) vertex_offset: u64,
    pub(super) index_buffer: vk::Buffer,
    pub(super) index_offset: u64,
}

#[derive(Clone, Copy)]
pub(super) enum Mesh2DPass {
    Depth,
    Depthless,
    Analytic,
}

pub(super) struct Mesh2DBindings<'a> {
    pub(super) batches: &'a [PreparedMesh2DBatch],
    pub(super) camera_descriptor_set: vk::DescriptorSet,
    pub(super) camera_dynamic_offsets: &'a [u32],
    pub(super) texture_descriptor_set: vk::DescriptorSet,
    pub(super) vertex_buffer: vk::Buffer,
    pub(super) index_buffer: vk::Buffer,
    pub(super) instance_buffer: vk::Buffer,
    pub(super) instance_offset: u64,
}

impl OutputPasses {
    fn has_compute_output(self) -> bool {
        self.cpu_nv12_descriptor_set.is_some()
            || self.cpu_yuv444p_descriptor_set.is_some()
            || self.video.is_some()
    }
}

impl<'a> CommandRecorder<'a> {
    pub(super) fn new(
        device: &'a ash::Device,
        command_buffer: vk::CommandBuffer,
        pipelines: &'a PipelineSet,
    ) -> Self {
        Self {
            device,
            command_buffer,
            pipelines,
        }
    }

    pub(super) unsafe fn record_geometry_uploads_2d(
        &self,
        uploads: &[GeometryUpload2D],
        buffers: GeometryUploadBuffers2D,
    ) {
        if uploads.is_empty() {
            return;
        }
        let vertex_copies: Vec<_> = uploads
            .iter()
            .map(|upload| vk::BufferCopy {
                src_offset: buffers.vertex_staging_base + upload.staging_vertex_offset,
                dst_offset: upload.device_vertex_offset,
                size: std::mem::size_of_val(upload.geometry.vertices()) as u64,
            })
            .collect();
        let index_copies: Vec<_> = uploads
            .iter()
            .map(|upload| vk::BufferCopy {
                src_offset: buffers.index_staging_base + upload.staging_index_offset,
                dst_offset: upload.device_index_offset,
                size: std::mem::size_of_val(upload.geometry.indices()) as u64,
            })
            .collect();
        unsafe {
            self.device.cmd_copy_buffer(
                self.command_buffer,
                buffers.vertex_staging,
                buffers.vertex_device,
                &vertex_copies,
            );
            self.device.cmd_copy_buffer(
                self.command_buffer,
                buffers.index_staging,
                buffers.index_device,
                &index_copies,
            );

            let barriers = [
                vk::BufferMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COPY)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT)
                    .dst_access_mask(vk::AccessFlags2::VERTEX_ATTRIBUTE_READ)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .buffer(buffers.vertex_device)
                    .offset(0)
                    .size(vk::WHOLE_SIZE),
                vk::BufferMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COPY)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::INDEX_INPUT)
                    .dst_access_mask(vk::AccessFlags2::INDEX_READ)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .buffer(buffers.index_device)
                    .offset(0)
                    .size(vk::WHOLE_SIZE),
            ];
            self.device.cmd_pipeline_barrier2(
                self.command_buffer,
                &vk::DependencyInfo::default().buffer_memory_barriers(&barriers),
            );
        }
    }

    pub(super) unsafe fn record_meshes_3d(&self, pass: Mesh3DPass, bindings: Mesh3DBindings<'_>) {
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

    pub(super) unsafe fn record_meshes_2d(&self, pass: Mesh2DPass, bindings: Mesh2DBindings<'_>) {
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

    pub(super) unsafe fn record_empty_frame(
        &self,
        targets: &mut RenderTargetSet,
        background_color: [f32; 4],
    ) {
        unsafe {
            transition_image(
                self.device,
                self.command_buffer,
                targets.texture.vk_image,
                vk::ImageAspectFlags::COLOR,
                &mut targets.texture_state,
                TrackedImageState {
                    layout: vk::ImageLayout::GENERAL,
                    stage: vk::PipelineStageFlags2::CLEAR,
                    access: vk::AccessFlags2::TRANSFER_WRITE,
                },
            );
            self.device.cmd_clear_color_image(
                self.command_buffer,
                targets.texture.vk_image,
                vk::ImageLayout::GENERAL,
                &vk::ClearColorValue {
                    float32: background_color,
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
    }

    pub(super) unsafe fn record_sdf(
        &self,
        targets: &mut RenderTargetSet,
        dynamic_offsets: &[u32],
        width: u32,
        height: u32,
    ) {
        let compute_write = TrackedImageState {
            layout: vk::ImageLayout::GENERAL,
            stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
            access: vk::AccessFlags2::SHADER_WRITE,
        };
        let compute_read = TrackedImageState {
            layout: vk::ImageLayout::GENERAL,
            stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
            access: vk::AccessFlags2::SHADER_READ,
        };
        unsafe {
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
                    self.device,
                    self.command_buffer,
                    image,
                    vk::ImageAspectFlags::COLOR,
                    state,
                    compute_write,
                );
            }
            self.device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipelines.compute_pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipelines.compute_pipeline_layout,
                0,
                std::slice::from_ref(&targets.compute_descriptor_set),
                dynamic_offsets,
            );
            self.device.cmd_dispatch(
                self.command_buffer,
                (width + 15) / 16,
                (height + 15) / 16,
                1,
            );
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
                    self.device,
                    self.command_buffer,
                    image,
                    vk::ImageAspectFlags::COLOR,
                    state,
                    compute_read,
                );
            }
        }
    }

    pub(super) unsafe fn record_bloom(&self, targets: &mut RenderTargetSet, enabled: bool) {
        let compute_read = TrackedImageState {
            layout: vk::ImageLayout::GENERAL,
            stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
            access: vk::AccessFlags2::SHADER_READ,
        };
        let compute_write = TrackedImageState {
            layout: vk::ImageLayout::GENERAL,
            stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
            access: vk::AccessFlags2::SHADER_WRITE,
        };

        if enabled {
            unsafe {
                transition_image(
                    self.device,
                    self.command_buffer,
                    targets.resolved_texture.vk_image,
                    vk::ImageAspectFlags::COLOR,
                    &mut targets.resolved_texture_state,
                    compute_read,
                );
                transition_image(
                    self.device,
                    self.command_buffer,
                    targets.bloom_ping.vk_image,
                    vk::ImageAspectFlags::COLOR,
                    &mut targets.bloom_ping_state,
                    compute_write,
                );
                self.record_compute_dispatch(
                    self.pipelines.bloom_extract_pipeline,
                    self.pipelines.bloom_pipeline_layout,
                    targets.bloom_descriptor_sets[0],
                    targets.bloom_ping.width,
                    targets.bloom_ping.height,
                );

                transition_image(
                    self.device,
                    self.command_buffer,
                    targets.bloom_ping.vk_image,
                    vk::ImageAspectFlags::COLOR,
                    &mut targets.bloom_ping_state,
                    compute_read,
                );
                transition_image(
                    self.device,
                    self.command_buffer,
                    targets.bloom_pong.vk_image,
                    vk::ImageAspectFlags::COLOR,
                    &mut targets.bloom_pong_state,
                    compute_write,
                );
                self.record_compute_dispatch(
                    self.pipelines.bloom_horizontal_pipeline,
                    self.pipelines.bloom_pipeline_layout,
                    targets.bloom_descriptor_sets[1],
                    targets.bloom_pong.width,
                    targets.bloom_pong.height,
                );

                transition_image(
                    self.device,
                    self.command_buffer,
                    targets.bloom_pong.vk_image,
                    vk::ImageAspectFlags::COLOR,
                    &mut targets.bloom_pong_state,
                    compute_read,
                );
                transition_image(
                    self.device,
                    self.command_buffer,
                    targets.bloom_ping.vk_image,
                    vk::ImageAspectFlags::COLOR,
                    &mut targets.bloom_ping_state,
                    compute_write,
                );
                self.record_compute_dispatch(
                    self.pipelines.bloom_vertical_pipeline,
                    self.pipelines.bloom_pipeline_layout,
                    targets.bloom_descriptor_sets[2],
                    targets.bloom_ping.width,
                    targets.bloom_ping.height,
                );
                transition_image(
                    self.device,
                    self.command_buffer,
                    targets.bloom_ping.vk_image,
                    vk::ImageAspectFlags::COLOR,
                    &mut targets.bloom_ping_state,
                    compute_read,
                );
            }
            targets.bloom_contains_data = true;
            return;
        }

        if !targets.bloom_contains_data
            && targets.bloom_ping_state.layout != vk::ImageLayout::UNDEFINED
        {
            return;
        }

        let clear_write = TrackedImageState {
            layout: vk::ImageLayout::GENERAL,
            stage: vk::PipelineStageFlags2::CLEAR,
            access: vk::AccessFlags2::TRANSFER_WRITE,
        };
        unsafe {
            transition_image(
                self.device,
                self.command_buffer,
                targets.bloom_ping.vk_image,
                vk::ImageAspectFlags::COLOR,
                &mut targets.bloom_ping_state,
                clear_write,
            );
            self.device.cmd_clear_color_image(
                self.command_buffer,
                targets.bloom_ping.vk_image,
                vk::ImageLayout::GENERAL,
                &vk::ClearColorValue { float32: [0.0; 4] },
                &[vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                }],
            );
            transition_image(
                self.device,
                self.command_buffer,
                targets.bloom_ping.vk_image,
                vk::ImageAspectFlags::COLOR,
                &mut targets.bloom_ping_state,
                compute_read,
            );
        }
        targets.bloom_contains_data = false;
    }

    pub(super) unsafe fn record_surface_composite(
        &self,
        targets: &mut RenderTargetSet,
        has_overlay: bool,
        width: u32,
        height: u32,
        raster_scale: u32,
    ) {
        let compute_read = TrackedImageState {
            layout: vk::ImageLayout::GENERAL,
            stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
            access: vk::AccessFlags2::SHADER_READ,
        };
        let compute_write = TrackedImageState {
            layout: vk::ImageLayout::GENERAL,
            stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
            access: vk::AccessFlags2::SHADER_WRITE,
        };
        unsafe {
            transition_image(
                self.device,
                self.command_buffer,
                targets.surface_hdr.vk_image,
                vk::ImageAspectFlags::COLOR,
                &mut targets.surface_hdr_state,
                compute_read,
            );
            if has_overlay {
                transition_image(
                    self.device,
                    self.command_buffer,
                    targets.overlay_hdr.vk_image,
                    vk::ImageAspectFlags::COLOR,
                    &mut targets.overlay_hdr_state,
                    compute_read,
                );
            }
            transition_image(
                self.device,
                self.command_buffer,
                targets.resolved_texture.vk_image,
                vk::ImageAspectFlags::COLOR,
                &mut targets.resolved_texture_state,
                compute_write,
            );
            self.record_compute_dispatch(
                if has_overlay {
                    self.pipelines.surface_overlay_pipeline
                } else {
                    self.pipelines.surface_copy_pipeline
                },
                self.pipelines.surface_composite_pipeline_layout,
                targets.surface_composite_descriptor_set,
                width * raster_scale,
                height * raster_scale,
            );
        }
        targets.resolved_texture_state = compute_write;
    }

    pub(super) unsafe fn record_tone_map(
        &self,
        targets: &mut RenderTargetSet,
        width: u32,
        height: u32,
    ) {
        let compute_read = TrackedImageState {
            layout: vk::ImageLayout::GENERAL,
            stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
            access: vk::AccessFlags2::SHADER_READ,
        };
        let compute_write = TrackedImageState {
            layout: vk::ImageLayout::GENERAL,
            stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
            access: vk::AccessFlags2::SHADER_WRITE,
        };
        unsafe {
            transition_image(
                self.device,
                self.command_buffer,
                targets.resolved_texture.vk_image,
                vk::ImageAspectFlags::COLOR,
                &mut targets.resolved_texture_state,
                compute_read,
            );
            transition_image(
                self.device,
                self.command_buffer,
                targets.texture.vk_image,
                vk::ImageAspectFlags::COLOR,
                &mut targets.texture_state,
                compute_write,
            );
            self.record_compute_dispatch(
                self.pipelines.downsample_pipeline,
                self.pipelines.composite_pipeline_layout,
                targets.composite_descriptor_set,
                width,
                height,
            );
        }
        targets.texture_state = compute_write;
    }

    pub(super) unsafe fn record_outputs(
        &self,
        targets: &mut RenderTargetSet,
        outputs: OutputPasses,
    ) {
        let compute_read = TrackedImageState {
            layout: vk::ImageLayout::GENERAL,
            stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
            access: vk::AccessFlags2::SHADER_READ,
        };
        if outputs.has_compute_output() {
            unsafe {
                if outputs.fused_video_downsample {
                    transition_image(
                        self.device,
                        self.command_buffer,
                        targets.resolved_texture.vk_image,
                        vk::ImageAspectFlags::COLOR,
                        &mut targets.resolved_texture_state,
                        compute_read,
                    );
                } else {
                    transition_image(
                        self.device,
                        self.command_buffer,
                        targets.texture.vk_image,
                        vk::ImageAspectFlags::COLOR,
                        &mut targets.texture_state,
                        compute_read,
                    );
                }
            }
        }

        if let Some(descriptor_set) = outputs.cpu_nv12_descriptor_set {
            unsafe {
                self.record_compute_dispatch(
                    self.pipelines.nv12_pipeline,
                    self.pipelines.nv12_pipeline_layout,
                    descriptor_set,
                    outputs.width / 4,
                    outputs.height / 2,
                );
            }
        }

        if let Some(descriptor_set) = outputs.cpu_yuv444p_descriptor_set {
            unsafe {
                self.record_compute_dispatch(
                    self.pipelines.yuv444p_pipeline,
                    self.pipelines.nv12_pipeline_layout,
                    descriptor_set,
                    outputs.width / 4,
                    outputs.height,
                );
            }
        }

        if let Some(video) = outputs.video {
            let barrier = vk::ImageMemoryBarrier {
                s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
                old_layout: video.current_layout,
                new_layout: vk::ImageLayout::GENERAL,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image: video.image,
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
            let (pipeline, dispatch_width, dispatch_height) = if outputs.fused_video_downsample {
                (
                    self.pipelines.video_nv12_downsample_pipeline,
                    outputs.width / 2,
                    outputs.height / 2,
                )
            } else {
                (
                    self.pipelines.video_nv12_pipeline,
                    outputs.width,
                    outputs.height,
                )
            };
            unsafe {
                self.device.cmd_pipeline_barrier(
                    self.command_buffer,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    std::slice::from_ref(&barrier),
                );
                self.record_compute_dispatch(
                    pipeline,
                    self.pipelines.video_nv12_pipeline_layout,
                    video.descriptor_set,
                    dispatch_width,
                    dispatch_height,
                );
            }
        }

        if let Some(rgba_buffer) = outputs.rgba_buffer {
            unsafe {
                transition_image(
                    self.device,
                    self.command_buffer,
                    targets.texture.vk_image,
                    vk::ImageAspectFlags::COLOR,
                    &mut targets.texture_state,
                    TrackedImageState {
                        layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        stage: vk::PipelineStageFlags2::COPY,
                        access: vk::AccessFlags2::TRANSFER_READ,
                    },
                );
                self.device.cmd_copy_image_to_buffer(
                    self.command_buffer,
                    targets.texture.vk_image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    rgba_buffer,
                    &[vk::BufferImageCopy {
                        buffer_offset: 0,
                        buffer_row_length: outputs.rgba_padded_bytes_per_row / 4,
                        buffer_image_height: outputs.height,
                        image_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                        image_extent: vk::Extent3D {
                            width: outputs.width,
                            height: outputs.height,
                            depth: 1,
                        },
                    }],
                );
            }
        }
    }

    unsafe fn record_compute_dispatch(
        &self,
        pipeline: vk::Pipeline,
        layout: vk::PipelineLayout,
        descriptor_set: vk::DescriptorSet,
        width: u32,
        height: u32,
    ) {
        unsafe {
            self.device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                layout,
                0,
                std::slice::from_ref(&descriptor_set),
                &[],
            );
            self.device.cmd_dispatch(
                self.command_buffer,
                (width + 15) / 16,
                (height + 15) / 16,
                1,
            );
        }
    }

    unsafe fn record_mesh_3d_draw(&self, draw: &Mesh3DDraw) {
        unsafe {
            self.device.cmd_draw_indexed(
                self.command_buffer,
                draw.index_count,
                1,
                draw.first_index,
                0,
                draw.material_index,
            );
        }
    }
}
