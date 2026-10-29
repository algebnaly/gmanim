use ash::vk;

use super::super::targets::RenderTargetSet;
use super::CommandRecorder;

impl<'a> CommandRecorder<'a> {
    pub(in crate::vulkan::renderer) unsafe fn record_surface_resolve(
        &self,
        targets: &RenderTargetSet,
        dynamic_offsets: &[u32],
        extent: vk::Extent2D,
    ) {
        unsafe {
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
                extent.width.div_ceil(16),
                extent.height.div_ceil(16),
                1,
            );
        }
    }

    pub(in crate::vulkan::renderer) unsafe fn record_surface_lighting(
        &self,
        targets: &RenderTargetSet,
        dynamic_offsets: &[u32],
        extent: vk::Extent2D,
    ) {
        unsafe {
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
                extent.width.div_ceil(16),
                extent.height.div_ceil(16),
                1,
            );
        }
    }

    pub(in crate::vulkan::renderer) unsafe fn record_bloom_extract(
        &self,
        targets: &RenderTargetSet,
    ) {
        unsafe {
            self.record_compute_dispatch(
                self.pipelines.bloom_extract_pipeline,
                self.pipelines.bloom_pipeline_layout,
                targets.bloom_descriptor_sets[0],
                targets.bloom_ping.width,
                targets.bloom_ping.height,
            );
        }
    }

    pub(in crate::vulkan::renderer) unsafe fn record_bloom_horizontal(
        &self,
        targets: &RenderTargetSet,
    ) {
        unsafe {
            self.record_compute_dispatch(
                self.pipelines.bloom_horizontal_pipeline,
                self.pipelines.bloom_pipeline_layout,
                targets.bloom_descriptor_sets[1],
                targets.bloom_pong.width,
                targets.bloom_pong.height,
            );
        }
    }

    pub(in crate::vulkan::renderer) unsafe fn record_bloom_vertical(
        &self,
        targets: &mut RenderTargetSet,
    ) {
        unsafe {
            self.record_compute_dispatch(
                self.pipelines.bloom_vertical_pipeline,
                self.pipelines.bloom_pipeline_layout,
                targets.bloom_descriptor_sets[2],
                targets.bloom_ping.width,
                targets.bloom_ping.height,
            );
        }
        targets.bloom_contains_data = true;
    }

    pub(in crate::vulkan::renderer) unsafe fn record_bloom_clear(
        &self,
        targets: &mut RenderTargetSet,
    ) {
        unsafe {
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
        }
        targets.bloom_contains_data = false;
    }

    pub(in crate::vulkan::renderer) unsafe fn record_surface_composite(
        &self,
        targets: &RenderTargetSet,
        has_overlay: bool,
        width: u32,
        height: u32,
        raster_scale: u32,
    ) {
        unsafe {
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
    }

    pub(in crate::vulkan::renderer) unsafe fn record_tone_map(
        &self,
        targets: &RenderTargetSet,
        width: u32,
        height: u32,
    ) {
        unsafe {
            self.record_compute_dispatch(
                self.pipelines.downsample_pipeline,
                self.pipelines.composite_pipeline_layout,
                targets.composite_descriptor_set,
                width,
                height,
            );
        }
    }
}
