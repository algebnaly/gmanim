use ash::vk;

use super::super::frame::{TrackedImageState, transition_image};
use super::super::targets::RenderTargetSet;
use super::CommandRecorder;

impl<'a> CommandRecorder<'a> {
    pub(in crate::vulkan::renderer) unsafe fn record_empty_frame(
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

    pub(in crate::vulkan::renderer) unsafe fn record_sdf(
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
                width.div_ceil(16),
                height.div_ceil(16),
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
}
