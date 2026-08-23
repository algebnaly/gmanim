use ash::vk;
use std::sync::Arc;

use crate::mobjects::mesh_2d::Vertex2D;
use crate::mobjects::mesh_3d::{SurfaceMaterial, Vertex};
use crate::vulkan::context::VulkanContext;

use super::frame::RENDER_FRAME_COUNT;
use super::mesh_2d::{GeometryUpload2D, Instance2D};
use super::prepared_frame::GpuSdfPrimitive;
use super::{
    Buffer, CameraUniform, CameraUniform2D, MAX_SURFACE_MATERIALS, MaterialData3D, Nv12Constants,
    ToneMapConstants, align_up,
};

pub(super) struct FrameBuffers {
    pub(super) vertex: Buffer,
    pub(super) index: Buffer,
    pub(super) camera: Buffer,
    pub(super) material_3d: Buffer,
    pub(super) primitive: Buffer,
    pub(super) nv12_constants: Buffer,
    pub(super) vertex_2d: Buffer,
    pub(super) index_2d: Buffer,
    pub(super) vertex_staging_2d: Buffer,
    pub(super) index_staging_2d: Buffer,
    pub(super) instance_2d: Buffer,
    pub(super) camera_2d: Buffer,
    pub(super) tone_map_factor: Buffer,
    pub(super) strides: FrameBufferStrides,
    static_vertex_2d_capacity: u64,
    static_index_2d_capacity: u64,
    material_scratch: Vec<MaterialData3D>,
}

#[derive(Clone, Copy)]
pub(super) struct FrameBufferStrides {
    pub(super) vertex: u64,
    pub(super) index: u64,
    pub(super) camera: u64,
    pub(super) material_3d: u64,
    pub(super) primitive: u64,
    pub(super) vertex_staging_2d: u64,
    pub(super) index_staging_2d: u64,
    pub(super) instance_2d: u64,
    pub(super) camera_2d: u64,
    pub(super) tone_map_factor: u64,
}

pub(super) struct FrameUpload<'a> {
    pub(super) camera: &'a CameraUniform,
    pub(super) camera_2d: &'a CameraUniform2D,
    pub(super) primitives: &'a [GpuSdfPrimitive],
    pub(super) materials: &'a [SurfaceMaterial],
    pub(super) mesh_vertices: &'a [Vertex],
    pub(super) mesh_indices: &'a [u32],
    pub(super) geometry_2d: &'a [GeometryUpload2D],
    pub(super) instances_2d: &'a [Instance2D],
    pub(super) tone_map_factor: u32,
}

#[derive(Clone, Copy)]
pub(super) struct UploadedFrame {
    pub(super) vertex_offset: u64,
    pub(super) index_offset: u64,
    pub(super) vertex_staging_2d_offset: u64,
    pub(super) index_staging_2d_offset: u64,
    pub(super) instance_2d_offset: u64,
    pub(super) compute_dynamic_offsets: [u32; 2],
    pub(super) surface_dynamic_offsets: [u32; 2],
    pub(super) raster_dynamic_offsets: [u32; 2],
    pub(super) raster_2d_dynamic_offsets: [u32; 1],
}

impl FrameBuffers {
    pub(super) fn new(ctx: &Arc<VulkanContext>) -> Self {
        let limits = unsafe {
            ctx.instance
                .get_physical_device_properties(ctx.physical_device)
                .limits
        };
        let uniform_alignment = limits.min_uniform_buffer_offset_alignment.max(1);
        let storage_alignment = limits.min_storage_buffer_offset_alignment.max(1);
        let static_vertex_2d_capacity = (std::mem::size_of::<Vertex2D>() * 1_000_000) as u64;
        let static_index_2d_capacity = (std::mem::size_of::<u32>() * 3_000_000) as u64;
        let strides = FrameBufferStrides {
            vertex: (std::mem::size_of::<Vertex>() * 1_000_000) as u64,
            index: (std::mem::size_of::<u32>() * 3_000_000) as u64,
            camera: align_up(
                std::mem::size_of::<CameraUniform>() as u64,
                uniform_alignment,
            ),
            material_3d: align_up(
                (std::mem::size_of::<MaterialData3D>() * MAX_SURFACE_MATERIALS) as u64,
                storage_alignment,
            ),
            primitive: align_up(
                (std::mem::size_of::<GpuSdfPrimitive>() * 10_000) as u64,
                storage_alignment,
            ),
            vertex_staging_2d: static_vertex_2d_capacity,
            index_staging_2d: static_index_2d_capacity,
            instance_2d: (std::mem::size_of::<Instance2D>() * 100_000) as u64,
            camera_2d: align_up(
                std::mem::size_of::<CameraUniform2D>() as u64,
                uniform_alignment,
            ),
            tone_map_factor: align_up(
                std::mem::size_of::<ToneMapConstants>() as u64,
                uniform_alignment,
            ),
        };
        let frame_count = RENDER_FRAME_COUNT as u64;
        Self {
            vertex: Buffer::new(
                ctx,
                strides.vertex * frame_count,
                vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                gpu_allocator::MemoryLocation::CpuToGpu,
            ),
            index: Buffer::new(
                ctx,
                strides.index * frame_count,
                vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                gpu_allocator::MemoryLocation::CpuToGpu,
            ),
            camera: Buffer::new(
                ctx,
                strides.camera * frame_count,
                vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                gpu_allocator::MemoryLocation::CpuToGpu,
            ),
            material_3d: Buffer::new(
                ctx,
                strides.material_3d * frame_count,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                gpu_allocator::MemoryLocation::CpuToGpu,
            ),
            primitive: Buffer::new(
                ctx,
                strides.primitive * frame_count,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                gpu_allocator::MemoryLocation::CpuToGpu,
            ),
            nv12_constants: Buffer::new(
                ctx,
                std::mem::size_of::<Nv12Constants>() as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                gpu_allocator::MemoryLocation::CpuToGpu,
            ),
            vertex_2d: Buffer::new(
                ctx,
                static_vertex_2d_capacity + strides.vertex_staging_2d * frame_count,
                vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                gpu_allocator::MemoryLocation::GpuOnly,
            ),
            index_2d: Buffer::new(
                ctx,
                static_index_2d_capacity + strides.index_staging_2d * frame_count,
                vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                gpu_allocator::MemoryLocation::GpuOnly,
            ),
            vertex_staging_2d: Buffer::new(
                ctx,
                strides.vertex_staging_2d * frame_count,
                vk::BufferUsageFlags::TRANSFER_SRC,
                gpu_allocator::MemoryLocation::CpuToGpu,
            ),
            index_staging_2d: Buffer::new(
                ctx,
                strides.index_staging_2d * frame_count,
                vk::BufferUsageFlags::TRANSFER_SRC,
                gpu_allocator::MemoryLocation::CpuToGpu,
            ),
            instance_2d: Buffer::new(
                ctx,
                strides.instance_2d * frame_count,
                vk::BufferUsageFlags::VERTEX_BUFFER,
                gpu_allocator::MemoryLocation::CpuToGpu,
            ),
            camera_2d: Buffer::new(
                ctx,
                strides.camera_2d * frame_count,
                vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                gpu_allocator::MemoryLocation::CpuToGpu,
            ),
            tone_map_factor: Buffer::new(
                ctx,
                strides.tone_map_factor * frame_count,
                vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                gpu_allocator::MemoryLocation::CpuToGpu,
            ),
            strides,
            static_vertex_2d_capacity,
            static_index_2d_capacity,
            material_scratch: Vec::with_capacity(MAX_SURFACE_MATERIALS),
        }
    }

    pub(super) fn static_mesh_2d_capacities(&self) -> (u64, u64) {
        (
            self.static_vertex_2d_capacity,
            self.static_index_2d_capacity,
        )
    }

    pub(super) fn upload(&mut self, frame_index: usize, upload: FrameUpload<'_>) -> UploadedFrame {
        let frame_index = frame_index as u64;
        let vertex_offset = self.strides.vertex * frame_index;
        let index_offset = self.strides.index * frame_index;
        let camera_offset = self.strides.camera * frame_index;
        let material_offset = self.strides.material_3d * frame_index;
        let primitive_offset = self.strides.primitive * frame_index;
        let vertex_staging_2d_offset = self.strides.vertex_staging_2d * frame_index;
        let index_staging_2d_offset = self.strides.index_staging_2d * frame_index;
        let instance_2d_offset = self.strides.instance_2d * frame_index;
        let camera_2d_offset = self.strides.camera_2d * frame_index;

        self.camera
            .write_bytes(camera_offset, bytemuck::bytes_of(upload.camera));
        self.camera_2d
            .write_bytes(camera_2d_offset, bytemuck::bytes_of(upload.camera_2d));
        self.tone_map_factor.write_bytes(
            self.strides.tone_map_factor * frame_index,
            bytemuck::bytes_of(&ToneMapConstants {
                factor: upload.tone_map_factor,
                _padding: [0; 3],
            }),
        );
        if !upload.materials.is_empty() {
            assert!(
                upload.materials.len() <= MAX_SURFACE_MATERIALS,
                "3D material count exceeds {MAX_SURFACE_MATERIALS}"
            );
            self.material_scratch.clear();
            self.material_scratch
                .extend(upload.materials.iter().copied().map(MaterialData3D::from));
            self.material_3d.write_bytes(
                material_offset,
                bytemuck::cast_slice(&self.material_scratch),
            );
        }
        Self::write_slice(
            &self.primitive,
            primitive_offset,
            self.strides.primitive,
            upload.primitives,
        );
        Self::write_slice(
            &self.vertex,
            vertex_offset,
            self.strides.vertex,
            upload.mesh_vertices,
        );
        Self::write_slice(
            &self.index,
            index_offset,
            self.strides.index,
            upload.mesh_indices,
        );
        for geometry in upload.geometry_2d {
            self.vertex_staging_2d.write_bytes(
                vertex_staging_2d_offset + geometry.staging_vertex_offset,
                bytemuck::cast_slice(geometry.geometry.vertices()),
            );
            self.index_staging_2d.write_bytes(
                index_staging_2d_offset + geometry.staging_index_offset,
                bytemuck::cast_slice(geometry.geometry.indices()),
            );
        }
        if !upload.instances_2d.is_empty() {
            self.instance_2d.write_bytes(
                instance_2d_offset,
                bytemuck::cast_slice(upload.instances_2d),
            );
        }

        UploadedFrame {
            vertex_offset,
            index_offset,
            vertex_staging_2d_offset,
            index_staging_2d_offset,
            instance_2d_offset,
            compute_dynamic_offsets: [camera_offset as u32, primitive_offset as u32],
            surface_dynamic_offsets: [camera_offset as u32, material_offset as u32],
            raster_dynamic_offsets: [camera_offset as u32, material_offset as u32],
            raster_2d_dynamic_offsets: [camera_2d_offset as u32],
        }
    }

    fn write_slice<T: bytemuck::NoUninit>(
        buffer: &Buffer,
        offset: u64,
        capacity: u64,
        values: &[T],
    ) {
        if values.is_empty() {
            return;
        }
        let bytes = bytemuck::cast_slice(values);
        let len = (capacity as usize).min(bytes.len());
        buffer.write_bytes(offset, &bytes[..len]);
    }
}
