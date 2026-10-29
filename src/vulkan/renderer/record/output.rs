use ash::vk;

use super::CommandRecorder;

impl<'a> CommandRecorder<'a> {
    pub(in crate::vulkan::renderer) unsafe fn record_output_cpu_nv12(
        &self,
        descriptor_set: vk::DescriptorSet,
        width: u32,
        height: u32,
    ) {
        unsafe {
            self.record_compute_dispatch(
                self.pipelines.nv12_pipeline,
                self.pipelines.nv12_pipeline_layout,
                descriptor_set,
                width / 4,
                height / 2,
            );
        }
    }

    pub(in crate::vulkan::renderer) unsafe fn record_output_cpu_yuv444p(
        &self,
        descriptor_set: vk::DescriptorSet,
        width: u32,
        height: u32,
    ) {
        unsafe {
            self.record_compute_dispatch(
                self.pipelines.yuv444p_pipeline,
                self.pipelines.nv12_pipeline_layout,
                descriptor_set,
                width / 4,
                height,
            );
        }
    }

    pub(in crate::vulkan::renderer) unsafe fn record_output_vulkan_video(
        &self,
        descriptor_set: vk::DescriptorSet,
        width: u32,
        height: u32,
        fused_downsample: bool,
    ) {
        let (pipeline, dispatch_width, dispatch_height) = if fused_downsample {
            (
                self.pipelines.video_nv12_downsample_pipeline,
                width / 2,
                height / 2,
            )
        } else {
            (self.pipelines.video_nv12_pipeline, width, height)
        };
        unsafe {
            self.record_compute_dispatch(
                pipeline,
                self.pipelines.video_nv12_pipeline_layout,
                descriptor_set,
                dispatch_width,
                dispatch_height,
            );
        }
    }

    pub(in crate::vulkan::renderer) unsafe fn record_copy_rgba_to_buffer(
        &self,
        src_image: vk::Image,
        dst_buffer: vk::Buffer,
        width: u32,
        height: u32,
        padded_bytes_per_row: u32,
    ) {
        unsafe {
            self.device.cmd_copy_image_to_buffer(
                self.command_buffer,
                src_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst_buffer,
                &[vk::BufferImageCopy {
                    buffer_offset: 0,
                    buffer_row_length: padded_bytes_per_row / 4,
                    buffer_image_height: height,
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                    image_extent: vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    },
                }],
            );
        }
    }
}
