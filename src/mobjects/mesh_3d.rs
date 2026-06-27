use crate::mobjects::{Draw, Mobject, Transform};
use crate::Color;
use crate::GMFloat;
use nalgebra::{Matrix4, Point3};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 4],
}

pub struct TriangleMesh3D {
    pub base: super::MobjectBase,
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub model_matrix: Matrix4<GMFloat>,
}

impl TriangleMesh3D {
    pub fn new(vertices: Vec<Vertex>, indices: Vec<u32>) -> Self {
        Self {
            base: super::MobjectBase::new("TriangleMesh3D"),
            vertices,
            indices,
            model_matrix: Matrix4::identity(),
        }
    }

    pub fn box_mesh(
        center: Point3<GMFloat>,
        size: nalgebra::Vector3<GMFloat>,
        color: Color,
    ) -> Self {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        let c = [
            color.r as f32 / 255.0,
            color.g as f32 / 255.0,
            color.b as f32 / 255.0,
            color.a as f32 / 255.0,
        ];
        let hw = size.x as f32;
        let hh = size.y as f32;
        let hd = size.z as f32;
        let cx = center.x as f32;
        let cy = center.y as f32;
        let cz = center.z as f32;

        let faces = [
            // front
            (
                [0.0, 0.0, 1.0],
                [
                    [cx - hw, cy - hh, cz + hd],
                    [cx + hw, cy - hh, cz + hd],
                    [cx + hw, cy + hh, cz + hd],
                    [cx - hw, cy + hh, cz + hd],
                ],
            ),
            // back
            (
                [0.0, 0.0, -1.0],
                [
                    [cx + hw, cy - hh, cz - hd],
                    [cx - hw, cy - hh, cz - hd],
                    [cx - hw, cy + hh, cz - hd],
                    [cx + hw, cy + hh, cz - hd],
                ],
            ),
            // top
            (
                [0.0, 1.0, 0.0],
                [
                    [cx - hw, cy + hh, cz + hd],
                    [cx + hw, cy + hh, cz + hd],
                    [cx + hw, cy + hh, cz - hd],
                    [cx - hw, cy + hh, cz - hd],
                ],
            ),
            // bottom
            (
                [0.0, -1.0, 0.0],
                [
                    [cx - hw, cy - hh, cz - hd],
                    [cx + hw, cy - hh, cz - hd],
                    [cx + hw, cy - hh, cz + hd],
                    [cx - hw, cy - hh, cz + hd],
                ],
            ),
            // right
            (
                [1.0, 0.0, 0.0],
                [
                    [cx + hw, cy - hh, cz + hd],
                    [cx + hw, cy - hh, cz - hd],
                    [cx + hw, cy + hh, cz - hd],
                    [cx + hw, cy + hh, cz + hd],
                ],
            ),
            // left
            (
                [-1.0, 0.0, 0.0],
                [
                    [cx - hw, cy - hh, cz - hd],
                    [cx - hw, cy - hh, cz + hd],
                    [cx - hw, cy + hh, cz + hd],
                    [cx - hw, cy + hh, cz - hd],
                ],
            ),
        ];

        for (normal, pos) in faces.iter() {
            let start_idx = vertices.len() as u32;
            for p in pos.iter() {
                vertices.push(Vertex {
                    position: *p,
                    normal: *normal,
                    color: c,
                });
            }
            indices.extend_from_slice(&[
                start_idx,
                start_idx + 1,
                start_idx + 2,
                start_idx,
                start_idx + 2,
                start_idx + 3,
            ]);
        }

        Self::new(vertices, indices)
    }

    pub fn uv_sphere(
        center: Point3<GMFloat>,
        radius: GMFloat,
        segments: u32,
        rings: u32,
        color: Color,
    ) -> Self {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        let c = [
            color.r as f32 / 255.0,
            color.g as f32 / 255.0,
            color.b as f32 / 255.0,
            color.a as f32 / 255.0,
        ];

        let cx = center.x as f32;
        let cy = center.y as f32;
        let cz = center.z as f32;
        let r = radius as f32;

        for i in 0..=rings {
            let v = i as f32 / rings as f32;
            let phi = v * std::f32::consts::PI;

            for j in 0..=segments {
                let u = j as f32 / segments as f32;
                let theta = u * std::f32::consts::PI * 2.0;

                let x = phi.sin() * theta.cos();
                let y = phi.cos();
                let z = phi.sin() * theta.sin();

                vertices.push(Vertex {
                    position: [cx + x * r, cy + y * r, cz + z * r],
                    normal: [x, y, z],
                    color: c,
                });
            }
        }

        for i in 0..rings {
            for j in 0..segments {
                let p1 = i * (segments + 1) + j;
                let p2 = p1 + segments + 1;

                indices.push(p1);
                indices.push(p2);
                indices.push(p1 + 1);

                indices.push(p1 + 1);
                indices.push(p2);
                indices.push(p2 + 1);
            }
        }

        Self::new(vertices, indices)
    }
}

impl Draw for TriangleMesh3D {
    fn draw(&self, _ctx: &mut crate::Context, _parent_matrix: nalgebra::Matrix4<crate::GMFloat>) {}
}

impl Mobject for TriangleMesh3D {
    fn submit_to_renderer(
        &self,
        visitor: &mut dyn crate::mobjects::RenderVisitor,
        parent_mat: nalgebra::Matrix4<crate::GMFloat>,
    ) {
        visitor.push_mesh_3d(self, parent_mat * self.base.model_matrix);
        let global_mat = parent_mat * self.base.model_matrix;
        for child in self.base.children.iter() {
            child.borrow().submit_to_renderer(visitor, global_mat);
        }
    }

    fn base(&self) -> &super::MobjectBase {
        &self.base
    }
    fn base_mut(&mut self) -> &mut super::MobjectBase {
        &mut self.base
    }
}
