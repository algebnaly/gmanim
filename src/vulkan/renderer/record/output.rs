use ash::vk;

use super::super::frame::{TrackedImageState, transition_image};
use super::super::targets::RenderTargetSet;
use super::{CommandRecorder, OutputPasses};

impl<'a> CommandRecorder<'a> {
    pub(in crate::vulkan::renderer) unsafe fn record_outputs(
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
}
