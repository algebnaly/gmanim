use crate::Color;
use crate::GMFloat;
use crate::mobjects::{Draw, Mobject};
use nalgebra::{Matrix4, Point3, Vector3};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 4],
    pub surface_coord: [f32; 3],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transmission3D {
    pub opacity: f32,
    pub fresnel_opacity: f32,
    pub absorption: [f32; 3],
    pub ior: f32,
    pub backface_opacity_scale: f32,
}

impl Default for Transmission3D {
    fn default() -> Self {
        Self {
            opacity: 0.04,
            fresnel_opacity: 0.55,
            absorption: [0.08, 0.025, 0.015],
            ior: 1.45,
            backface_opacity_scale: 0.7,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AlphaMode3D {
    Opaque,
    Blend(Transmission3D),
}

#[derive(Clone, Copy, Debug)]
pub struct SphericalGridMaterial {
    pub color: [f32; 4],
    pub longitude_count: f32,
    pub latitude_count: f32,
    pub line_width_pixels: f32,
    pub backface_intensity: f32,
}

impl Default for SphericalGridMaterial {
    fn default() -> Self {
        Self {
            color: [0.45, 0.68, 0.72, 0.65],
            longitude_count: 16.0,
            latitude_count: 12.0,
            line_width_pixels: 1.0,
            backface_intensity: 0.25,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SphericalPatchMaterial {
    pub directions: [[f32; 3]; 3],
    pub color: [f32; 4],
    pub edge_color: [f32; 4],
    pub edge_width_pixels: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct SurfaceMaterial {
    pub base_color: [f32; 4],
    pub emissive: [f32; 3],
    pub emissive_strength: f32,
    pub roughness: f32,
    pub metallic: f32,
    pub reflectance: f32,
    pub alpha_mode: AlphaMode3D,
    pub spherical_grid: Option<SphericalGridMaterial>,
    pub spherical_patch: Option<SphericalPatchMaterial>,
}

impl Default for SurfaceMaterial {
    fn default() -> Self {
        Self {
            base_color: [1.0; 4],
            emissive: [0.0; 3],
            emissive_strength: 0.0,
            roughness: 0.55,
            metallic: 0.0,
            reflectance: 0.5,
            alpha_mode: AlphaMode3D::Opaque,
            spherical_grid: None,
            spherical_patch: None,
        }
    }
}

pub struct TriangleMesh3D {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub model_matrix: Matrix4<GMFloat>,
    pub material: SurfaceMaterial,
}

impl TriangleMesh3D {
    pub fn new(vertices: Vec<Vertex>, indices: Vec<u32>) -> Self {
        let alpha_mode = if vertices.iter().any(|vertex| vertex.color[3] < 0.999) {
            AlphaMode3D::Blend(Transmission3D {
                opacity: 1.0,
                ..Default::default()
            })
        } else {
            AlphaMode3D::Opaque
        };
        Self {
            vertices,
            indices,
            model_matrix: Matrix4::identity(),
            material: SurfaceMaterial {
                alpha_mode,
                ..Default::default()
            },
        }
    }

    pub fn with_material(mut self, material: SurfaceMaterial) -> Self {
        self.material = material;
        self
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
                    surface_coord: *normal,
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
                    surface_coord: [x, y, z],
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

    pub fn spherical_triangle(
        center: Point3<GMFloat>,
        radius: GMFloat,
        corners: [Vector3<GMFloat>; 3],
        subdivisions: u32,
        color: Color,
    ) -> Self {
        assert!(subdivisions > 0, "subdivisions must be greater than zero");
        let corners = corners.map(|corner| corner.normalize());
        let center = center.cast::<f32>();
        let radius = radius as f32;
        let color = [
            color.r as f32 / 255.0,
            color.g as f32 / 255.0,
            color.b as f32 / 255.0,
            color.a as f32 / 255.0,
        ];
        let mut vertices = Vec::new();
        let mut rows = Vec::with_capacity(subdivisions as usize + 1);

        for i in 0..=subdivisions {
            let mut row = Vec::with_capacity((subdivisions - i + 1) as usize);
            for j in 0..=subdivisions - i {
                let weight_b = i as GMFloat / subdivisions as GMFloat;
                let weight_c = j as GMFloat / subdivisions as GMFloat;
                let weight_a = 1.0 - weight_b - weight_c;
                let direction =
                    (corners[0] * weight_a + corners[1] * weight_b + corners[2] * weight_c)
                        .normalize()
                        .cast::<f32>();
                row.push(vertices.len() as u32);
                vertices.push(Vertex {
                    position: [
                        center.x + direction.x * radius,
                        center.y + direction.y * radius,
                        center.z + direction.z * radius,
                    ],
                    normal: [direction.x, direction.y, direction.z],
                    color,
                    surface_coord: [direction.x, direction.y, direction.z],
                });
            }
            rows.push(row);
        }

        let mut indices = Vec::with_capacity((subdivisions * subdivisions * 3) as usize);
        for i in 0..subdivisions as usize {
            for j in 0..(subdivisions as usize - i) {
                indices.extend_from_slice(&[rows[i][j], rows[i + 1][j], rows[i][j + 1]]);
                if j + 1 < rows[i + 1].len() {
                    indices.extend_from_slice(&[
                        rows[i][j + 1],
                        rows[i + 1][j],
                        rows[i + 1][j + 1],
                    ]);
                }
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
                surface_coord: normal,
            });
            vertices.push(Vertex {
                position: [
                    end_f[0] + normal[0] * r,
                    end_f[1] + normal[1] * r,
                    end_f[2] + normal[2] * r,
                ],
                normal,
                color: c,
                surface_coord: normal,
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
            surface_coord: [-axis_f[0], -axis_f[1], -axis_f[2]],
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
                surface_coord: normal,
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
            surface_coord: axis_f,
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
                surface_coord: normal,
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
            surface_coord: axis_f,
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
                surface_coord: normal,
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
            surface_coord: [-axis_f[0], -axis_f[1], -axis_f[2]],
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
                surface_coord: normal,
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
    fn default_name(&self) -> &'static str {
        "TriangleMesh3D"
    }

    fn submit_to_renderer(
        &self,
        visitor: &mut dyn crate::mobjects::RenderVisitor,
        world_transform: nalgebra::Matrix4<crate::GMFloat>,
    ) {
        visitor.push_surface_3d(crate::mobjects::Surface3DSubmission {
            geometry: crate::mobjects::Geometry3DRef::Mesh(self),
            material: self.material,
            transform: world_transform,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spherical_triangle_vertices_stay_on_the_sphere() {
        let center = Point3::new(1.0, -2.0, 0.5);
        let radius = 3.25;
        let mesh = TriangleMesh3D::spherical_triangle(
            center,
            radius,
            [
                Vector3::new(1.0, 0.0, 0.0),
                Vector3::new(0.0, 1.0, 0.0),
                Vector3::new(0.0, 0.0, 1.0),
            ],
            8,
            Color::white(),
        );

        assert_eq!(mesh.vertices.len(), 45);
        assert_eq!(mesh.indices.len(), 8 * 8 * 3);
        for vertex in mesh.vertices {
            let point = Point3::new(vertex.position[0], vertex.position[1], vertex.position[2]);
            assert!(((point - center).norm() - radius).abs() < 1e-4);
        }
    }

    #[test]
    fn alpha_mesh_defaults_to_transparent_blending() {
        let mesh = TriangleMesh3D::uv_sphere(Point3::origin(), 1.0, 8, 4, Color::new(1, 2, 3, 127));
        assert!(matches!(mesh.material.alpha_mode, AlphaMode3D::Blend(_)));
    }
}
