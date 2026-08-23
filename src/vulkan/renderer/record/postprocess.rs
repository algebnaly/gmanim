use ash::vk;

use super::super::frame::{TrackedImageState, transition_image};
use super::super::targets::RenderTargetSet;
use super::CommandRecorder;

impl<'a> CommandRecorder<'a> {
    pub(in crate::vulkan::renderer) unsafe fn record_surface_compute(
        &self,
        targets: &mut RenderTargetSet,
        dynamic_offsets: &[u32],
        extent: vk::Extent2D,
    ) {
        let compute_write = TrackedImageState {
            layout: vk::ImageLayout::GENERAL,
            stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
            access: vk::AccessFlags2::SHADER_WRITE,
        };
        let compute_read = TrackedImageState {
            access: vk::AccessFlags2::SHADER_READ,
            ..compute_write
        };

        unsafe {
            self.transition_resolved_surface(targets, compute_write);
            self.device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipelines.surface_resolve_pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipelines.surface_resolve_pipeline_layout,
                0,
                std::slice::from_ref(&targets.surface_resolve_descriptor_set),
                dynamic_offsets,
            );
            self.device.cmd_dispatch(
                self.command_buffer,
                (extent.width + 15) / 16,
                (extent.height + 15) / 16,
                1,
            );

            self.transition_resolved_surface(targets, compute_read);
            transition_image(
                self.device,
                self.command_buffer,
                targets.surface_hdr.vk_image,
                vk::ImageAspectFlags::COLOR,
                &mut targets.surface_hdr_state,
                compute_write,
            );
            self.device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipelines.surface_lighting_pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipelines.surface_lighting_pipeline_layout,
                0,
                std::slice::from_ref(&targets.surface_lighting_descriptor_set),
                dynamic_offsets,
            );
            self.device.cmd_dispatch(
                self.command_buffer,
                (extent.width + 15) / 16,
                (extent.height + 15) / 16,
                1,
            );
        }
    }

    unsafe fn transition_resolved_surface(
        &self,
        targets: &mut RenderTargetSet,
        destination: TrackedImageState,
    ) {
        for (image, state) in [
            (
                targets.resolved_primary_normal_depth.vk_image,
                &mut targets.resolved_primary_normal_depth_state,
            ),
            (
                targets.resolved_primary_albedo_coverage.vk_image,
                &mut targets.resolved_primary_albedo_coverage_state,
            ),
            (
                targets.resolved_secondary_normal_depth.vk_image,
                &mut targets.resolved_secondary_normal_depth_state,
            ),
            (
                targets.resolved_secondary_albedo_coverage.vk_image,
                &mut targets.resolved_secondary_albedo_coverage_state,
            ),
            (
                targets.resolved_material_ids.vk_image,
                &mut targets.resolved_material_ids_state,
            ),
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

    pub(in crate::vulkan::renderer) unsafe fn record_bloom(
        &self,
        targets: &mut RenderTargetSet,
        enabled: bool,
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

    pub(in crate::vulkan::renderer) unsafe fn record_surface_composite(
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

    pub(in crate::vulkan::renderer) unsafe fn record_tone_map(
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
}
