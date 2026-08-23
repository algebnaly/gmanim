use ash::vk;

use super::super::mesh_2d::GeometryUpload2D;
use super::{CommandRecorder, GeometryUploadBuffers2D};

impl<'a> CommandRecorder<'a> {
    pub(in crate::vulkan::renderer) unsafe fn record_geometry_uploads_2d(
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
}
