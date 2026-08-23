use super::mesh_2d::{GeometryUpload2D, Instance2D};
use super::{
    Buffer, CameraUniform, CameraUniform2D, GpuSdfPrimitive, MAX_SURFACE_MATERIALS, MaterialData3D,
    SurfaceMaterial, ToneMapConstants, Vertex,
};

pub(super) struct FrameUploader<'a> {
    pub(super) vertex: &'a Buffer,
    pub(super) index: &'a Buffer,
    pub(super) camera: &'a Buffer,
    pub(super) material_3d: &'a Buffer,
    pub(super) primitive: &'a Buffer,
    pub(super) vertex_staging_2d: &'a Buffer,
    pub(super) index_staging_2d: &'a Buffer,
    pub(super) instance_2d: &'a Buffer,
    pub(super) camera_2d: &'a Buffer,
    pub(super) tone_map_factor: &'a Buffer,
    pub(super) strides: FrameBufferStrides,
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

impl FrameUploader<'_> {
    pub(super) fn upload(&self, frame_index: usize, upload: FrameUpload<'_>) -> UploadedFrame {
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
            let materials: Vec<_> = upload
                .materials
                .iter()
                .copied()
                .map(MaterialData3D::from)
                .collect();
            self.material_3d
                .write_bytes(material_offset, bytemuck::cast_slice(&materials));
        }
        Self::write_slice(
            self.primitive,
            primitive_offset,
            self.strides.primitive,
            upload.primitives,
        );
        Self::write_slice(
            self.vertex,
            vertex_offset,
            self.strides.vertex,
            upload.mesh_vertices,
        );
        Self::write_slice(
            self.index,
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
