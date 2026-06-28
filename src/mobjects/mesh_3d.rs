use crate::Color;
use crate::GMFloat;
use crate::mobjects::{Draw, Mobject, Transform};
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

    pub fn cylinder(
        start: Point3<GMFloat>,
        end: Point3<GMFloat>,
        radius: GMFloat,
        segments: u32,
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

        let axis = (end - start).normalize();
        let mut up = nalgebra::Vector3::new(0.0, 1.0, 0.0);
        if axis.cross(&up).norm() < 1e-5 {
            up = nalgebra::Vector3::new(1.0, 0.0, 0.0);
        }
        let u1 = axis.cross(&up).normalize();
        let u2 = axis.cross(&u1).normalize();
        let r = radius as f32;

        let start_f = [start.x as f32, start.y as f32, start.z as f32];
        let end_f = [end.x as f32, end.y as f32, end.z as f32];
        let axis_f = [axis.x as f32, axis.y as f32, axis.z as f32];

        // Tube vertices
        for i in 0..=segments {
            let theta = i as f32 * std::f32::consts::PI * 2.0 / segments as f32;
            let p = u1 * (theta.cos() as GMFloat) + u2 * (theta.sin() as GMFloat);
            let normal = [p.x as f32, p.y as f32, p.z as f32];

            vertices.push(Vertex {
                position: [
                    start_f[0] + normal[0] * r,
                    start_f[1] + normal[1] * r,
                    start_f[2] + normal[2] * r,
                ],
                normal,
                color: c,
            });
            vertices.push(Vertex {
                position: [
                    end_f[0] + normal[0] * r,
                    end_f[1] + normal[1] * r,
                    end_f[2] + normal[2] * r,
                ],
                normal,
                color: c,
            });
        }

        // Tube indices
        for i in 0..segments {
            let p1 = i * 2;
            let p2 = p1 + 1;
            let p3 = (i + 1) * 2;
            let p4 = p3 + 1;

            indices.extend_from_slice(&[p1, p2, p3, p3, p2, p4]);
        }

        // Caps
        let cap_start_idx = vertices.len() as u32;
        vertices.push(Vertex {
            position: start_f,
            normal: [-axis_f[0], -axis_f[1], -axis_f[2]],
            color: c,
        }); // Bottom center
        for i in 0..segments {
            let theta = i as f32 * std::f32::consts::PI * 2.0 / segments as f32;
            let p = u1 * (theta.cos() as GMFloat) + u2 * (theta.sin() as GMFloat);
            let normal = [-axis_f[0], -axis_f[1], -axis_f[2]];
            vertices.push(Vertex {
                position: [
                    start_f[0] + p.x as f32 * r,
                    start_f[1] + p.y as f32 * r,
                    start_f[2] + p.z as f32 * r,
                ],
                normal,
                color: c,
            });
        }
        for i in 0..segments {
            indices.extend_from_slice(&[
                cap_start_idx,
                cap_start_idx + 1 + i,
                cap_start_idx + 1 + (i + 1) % segments,
            ]);
        }

        let cap_end_idx = vertices.len() as u32;
        vertices.push(Vertex {
            position: end_f,
            normal: axis_f,
            color: c,
        }); // Top center
        for i in 0..segments {
            let theta = i as f32 * std::f32::consts::PI * 2.0 / segments as f32;
            let p = u1 * (theta.cos() as GMFloat) + u2 * (theta.sin() as GMFloat);
            let normal = axis_f;
            vertices.push(Vertex {
                position: [
                    end_f[0] + p.x as f32 * r,
                    end_f[1] + p.y as f32 * r,
                    end_f[2] + p.z as f32 * r,
                ],
                normal,
                color: c,
            });
        }
        // Notice the winding order is reversed for the top cap so it faces outward
        for i in 0..segments {
            indices.extend_from_slice(&[
                cap_end_idx,
                cap_end_idx + 1 + (i + 1) % segments,
                cap_end_idx + 1 + i,
            ]);
        }

        Self::new(vertices, indices)
    }

    pub fn cone(
        base_center: Point3<GMFloat>,
        tip: Point3<GMFloat>,
        radius: GMFloat,
        segments: u32,
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

        let axis = (tip - base_center).normalize();
        let mut up = nalgebra::Vector3::new(0.0, 1.0, 0.0);
        if axis.cross(&up).norm() < 1e-5 {
            up = nalgebra::Vector3::new(1.0, 0.0, 0.0);
        }
        let u1 = axis.cross(&up).normalize();
        let u2 = axis.cross(&u1).normalize();
        let r = radius as f32;

        let base_f = [
            base_center.x as f32,
            base_center.y as f32,
            base_center.z as f32,
        ];
        let tip_f = [tip.x as f32, tip.y as f32, tip.z as f32];
        let axis_f = [axis.x as f32, axis.y as f32, axis.z as f32];
        let height = (tip - base_center).norm() as f32;

        // Side vertices
        let tip_idx = vertices.len() as u32;
        vertices.push(Vertex {
            position: tip_f,
            normal: axis_f,
            color: c,
        });

        let base_start_idx = vertices.len() as u32;
        for i in 0..segments {
            let theta = i as f32 * std::f32::consts::PI * 2.0 / segments as f32;
            let p = u1 * (theta.cos() as GMFloat) + u2 * (theta.sin() as GMFloat);

            // Calculate normal for cone side
            // Normal points outwards and slightly upwards.
            // Component along radius is H, component along axis is R.
            let mut normal_vec = p * (height as GMFloat) + axis * (radius as GMFloat);
            normal_vec.normalize_mut();
            let normal = [
                normal_vec.x as f32,
                normal_vec.y as f32,
                normal_vec.z as f32,
            ];

            vertices.push(Vertex {
                position: [
                    base_f[0] + p.x as f32 * r,
                    base_f[1] + p.y as f32 * r,
                    base_f[2] + p.z as f32 * r,
                ],
                normal,
                color: c,
            });
        }

        for i in 0..segments {
            let next_i = (i + 1) % segments;
            indices.extend_from_slice(&[tip_idx, base_start_idx + next_i, base_start_idx + i]);
        }

        // Base cap
        let cap_center_idx = vertices.len() as u32;
        vertices.push(Vertex {
            position: base_f,
            normal: [-axis_f[0], -axis_f[1], -axis_f[2]],
            color: c,
        });

        let cap_edge_start = vertices.len() as u32;
        for i in 0..segments {
            let theta = i as f32 * std::f32::consts::PI * 2.0 / segments as f32;
            let p = u1 * (theta.cos() as GMFloat) + u2 * (theta.sin() as GMFloat);
            let normal = [-axis_f[0], -axis_f[1], -axis_f[2]];
            vertices.push(Vertex {
                position: [
                    base_f[0] + p.x as f32 * r,
                    base_f[1] + p.y as f32 * r,
                    base_f[2] + p.z as f32 * r,
                ],
                normal,
                color: c,
            });
        }
        for i in 0..segments {
            let next_i = (i + 1) % segments;
            indices.extend_from_slice(&[
                cap_center_idx,
                cap_edge_start + i,
                cap_edge_start + next_i,
            ]);
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
