use nalgebra::{Point3, Vector3};
use rayon::prelude::*;
use std::sync::Arc;

use crate::camera::Camera;
use crate::mobjects::{Draw, Mobject, Transform};
use crate::{Color, Context, GMFloat};

// ═══════════════════════════════════════════════════════════════════════════
// 3D Object Trait
// ═══════════════════════════════════════════════════════════════════════════

/// Represents an object defined by a Signed Distance Function.
pub trait Object3D: Sync + Send {
    /// Returns the shortest distance from point `p` to the surface of the object (Used by CPU rendering).
    fn distance(&self, p: &Point3<GMFloat>) -> GMFloat;
    /// Returns the color of the object at point `p`.
    fn color(&self, p: &Point3<GMFloat>) -> Color;

    /// Returns the GPU-compatible primitive data
    fn as_primitive_data(
        &self,
        global_mat: nalgebra::Matrix4<GMFloat>,
    ) -> crate::vulkan::renderer::PrimitiveData3D;
}

// ═══════════════════════════════════════════════════════════════════════════
// Primitive 3D Objects
// ═══════════════════════════════════════════════════════════════════════════

/// A 3D Sphere
pub struct Sphere3D {
    pub center: Point3<GMFloat>,
    pub radius: GMFloat,
    pub color: Color,
    pub model_matrix: nalgebra::Matrix4<GMFloat>,
}

impl Object3D for Sphere3D {
    fn distance(&self, p: &Point3<GMFloat>) -> GMFloat {
        (p - self.center).norm() - self.radius
    }
    fn color(&self, _p: &Point3<GMFloat>) -> Color {
        self.color
    }
    fn as_primitive_data(
        &self,
        global_mat: nalgebra::Matrix4<GMFloat>,
    ) -> crate::vulkan::renderer::PrimitiveData3D {
        let center = global_mat.transform_point(&self.center);
        crate::vulkan::renderer::PrimitiveData3D {
            color: [
                self.color.r as f32 / 255.0,
                self.color.g as f32 / 255.0,
                self.color.b as f32 / 255.0,
                self.color.a as f32 / 255.0,
            ],
            params: [
                center.x as f32,
                center.y as f32,
                center.z as f32,
                self.radius as f32,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            shape_type: 0,
            padding: [0; 3],
        }
    }
}

impl Transform for Sphere3D {
    fn get_model_matrix(&self) -> nalgebra::Matrix4<GMFloat> {
        self.model_matrix
    }
    fn set_model_matrix(&mut self, mat: nalgebra::Matrix4<GMFloat>) {
        self.model_matrix = mat;
    }
}
impl Draw for Sphere3D {
    fn draw(&self, _ctx: &mut Context, _parent_matrix: nalgebra::Matrix4<GMFloat>) {}
}
impl Mobject for Sphere3D {
    fn as_3d(&self) -> Option<&dyn crate::mobjects::object_3d::Object3D> {
        Some(self)
    }

    fn set_position(&mut self, pos: nalgebra::Point3<GMFloat>) {
        self.center = pos;
    }

    fn get_position(&self) -> nalgebra::Point3<GMFloat> {
        self.center
    }
}

/// A 3D Line Segment (Capsule)
pub struct LineSegment3D {
    pub a: Point3<GMFloat>,
    pub b: Point3<GMFloat>,
    pub radius: GMFloat,
    pub color: Color,
    pub model_matrix: nalgebra::Matrix4<GMFloat>,
}

impl Object3D for LineSegment3D {
    fn distance(&self, p: &Point3<GMFloat>) -> GMFloat {
        let pa = p - self.a;
        let ba = self.b - self.a;
        let h = (pa.dot(&ba) / ba.dot(&ba)).clamp(0.0, 1.0);
        (pa - ba * h).norm() - self.radius
    }
    fn color(&self, _p: &Point3<GMFloat>) -> Color {
        self.color
    }
    fn as_primitive_data(
        &self,
        global_mat: nalgebra::Matrix4<GMFloat>,
    ) -> crate::vulkan::renderer::PrimitiveData3D {
        let a = global_mat.transform_point(&self.a);
        let b = global_mat.transform_point(&self.b);
        crate::vulkan::renderer::PrimitiveData3D {
            color: [
                self.color.r as f32 / 255.0,
                self.color.g as f32 / 255.0,
                self.color.b as f32 / 255.0,
                self.color.a as f32 / 255.0,
            ],
            params: [
                a.x as f32,
                a.y as f32,
                a.z as f32,
                b.x as f32,
                b.y as f32,
                b.z as f32,
                self.radius as f32,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            shape_type: 1,
            padding: [0; 3],
        }
    }
}

impl Transform for LineSegment3D {
    fn get_model_matrix(&self) -> nalgebra::Matrix4<GMFloat> {
        self.model_matrix
    }
    fn set_model_matrix(&mut self, mat: nalgebra::Matrix4<GMFloat>) {
        self.model_matrix = mat;
    }
}
impl Draw for LineSegment3D {
    fn draw(&self, _ctx: &mut Context, _parent_matrix: nalgebra::Matrix4<GMFloat>) {}
}
impl Mobject for LineSegment3D {
    fn as_3d(&self) -> Option<&dyn crate::mobjects::object_3d::Object3D> {
        Some(self)
    }
}

/// Inigo Quilez capped cone SDF in local coords: axial from 0 (base) to h (tip), radial distance.
fn sd_capped_cone_local(axial: GMFloat, radial: GMFloat, h: GMFloat, r: GMFloat) -> GMFloat {
    let q = nalgebra::Vector2::new(radial, axial);
    let k1 = nalgebra::Vector2::new(r, h);
    let k2 = nalgebra::Vector2::new(r - h, h);
    let cap_r = if q.y < 0.0 { r } else { q.y * r / h };
    let ca = nalgebra::Vector2::new(q.x - q.x.min(cap_r), q.y.abs() - h);
    let t = ((k2 - q).dot(&k1) / k1.dot(&k1)).clamp(0.0, 1.0);
    let cb = q - k2 + k1 * t;
    let s = if cb.x < 0.0 && ca.y < 0.0 { -1.0 } else { 1.0 };
    s * ca.dot(&ca).min(cb.dot(&cb)).sqrt()
}

/// A 3D Arrow (Union of a line segment and a cone)
pub struct Arrow3D {
    pub start: Point3<GMFloat>,
    pub end: Point3<GMFloat>,
    pub shaft_radius: GMFloat,
    pub head_radius: GMFloat,
    pub head_length: GMFloat,
    pub color: Color,
    pub model_matrix: nalgebra::Matrix4<GMFloat>,
}

impl Object3D for Arrow3D {
    fn distance(&self, p: &Point3<GMFloat>) -> GMFloat {
        let ba = self.end - self.start;
        let len = ba.norm();
        if len < 0.0001 {
            return (p - self.start).norm() - self.shaft_radius;
        }
        let dir = ba / len;
        let head_base = self.end - dir * self.head_length;

        // 1. Shaft (Capsule from start to head_base)
        let pa_s = p - self.start;
        let ba_s = head_base - self.start;
        let h_s = (pa_s.dot(&ba_s) / ba_s.dot(&ba_s)).clamp(0.0, 1.0);
        let d_shaft = (pa_s - ba_s * h_s).norm() - self.shaft_radius;

        // 2. Head (Capped cone from head_base to end)
        let q = p - head_base;
        let x = q.dot(&dir);
        let cr = (q - dir * x).norm();
        let d_cone = sd_capped_cone_local(x, cr, self.head_length, self.head_radius);

        // 3. Union (Min)
        d_shaft.min(d_cone)
    }

    fn color(&self, _p: &Point3<GMFloat>) -> Color {
        self.color
    }

    fn as_primitive_data(
        &self,
        global_mat: nalgebra::Matrix4<GMFloat>,
    ) -> crate::vulkan::renderer::PrimitiveData3D {
        let start = global_mat.transform_point(&self.start);
        let end = global_mat.transform_point(&self.end);
        crate::vulkan::renderer::PrimitiveData3D {
            color: [
                self.color.r as f32 / 255.0,
                self.color.g as f32 / 255.0,
                self.color.b as f32 / 255.0,
                self.color.a as f32 / 255.0,
            ],
            params: [
                start.x as f32,
                start.y as f32,
                start.z as f32,
                end.x as f32,
                end.y as f32,
                end.z as f32,
                self.shaft_radius as f32,
                self.head_radius as f32,
                self.head_length as f32,
                0.0,
                0.0,
                0.0,
            ],
            shape_type: 2,
            padding: [0; 3],
        }
    }
}

impl Transform for Arrow3D {
    fn get_model_matrix(&self) -> nalgebra::Matrix4<GMFloat> {
        self.model_matrix
    }
    fn set_model_matrix(&mut self, mat: nalgebra::Matrix4<GMFloat>) {
        self.model_matrix = mat;
    }
}
impl Draw for Arrow3D {
    fn draw(&self, _ctx: &mut Context, _parent_matrix: nalgebra::Matrix4<GMFloat>) {}
}
impl Mobject for Arrow3D {
    fn as_3d(&self) -> Option<&dyn crate::mobjects::object_3d::Object3D> {
        Some(self)
    }
}

pub struct Box3DSdf {
    pub center: Point3<GMFloat>,
    pub size: Vector3<GMFloat>,
    pub x_axis: Vector3<GMFloat>,
    pub y_axis: Vector3<GMFloat>,
    pub z_axis: Vector3<GMFloat>,
    pub color: Color,
    pub model_matrix: nalgebra::Matrix4<GMFloat>,
}

impl Object3D for Box3DSdf {
    fn distance(&self, p: &Point3<GMFloat>) -> GMFloat {
        let pt = p - self.center;
        let local_p = Vector3::new(
            pt.dot(&self.x_axis),
            pt.dot(&self.y_axis),
            pt.dot(&self.z_axis),
        );
        let d = Vector3::new(
            local_p.x.abs() - self.size.x,
            local_p.y.abs() - self.size.y,
            local_p.z.abs() - self.size.z,
        );

        let d_max = Vector3::new(d.x.max(0.0), d.y.max(0.0), d.z.max(0.0));
        let max_comp = d.x.max(d.y).max(d.z).min(0.0);
        d_max.norm() + max_comp
    }

    fn color(&self, _p: &Point3<GMFloat>) -> Color {
        self.color
    }

    fn as_primitive_data(
        &self,
        global_mat: nalgebra::Matrix4<GMFloat>,
    ) -> crate::vulkan::renderer::PrimitiveData3D {
        let center = global_mat.transform_point(&self.center);
        let x_axis = global_mat.transform_vector(&self.x_axis);
        let y_axis = global_mat.transform_vector(&self.y_axis);
        let z_axis = global_mat.transform_vector(&self.z_axis);
        crate::vulkan::renderer::PrimitiveData3D {
            color: [
                self.color.r as f32 / 255.0,
                self.color.g as f32 / 255.0,
                self.color.b as f32 / 255.0,
                self.color.a as f32 / 255.0,
            ],
            params: [
                center.x as f32,
                center.y as f32,
                center.z as f32,
                self.size.x as f32,
                self.size.y as f32,
                self.size.z as f32,
                x_axis.x as f32,
                x_axis.y as f32,
                x_axis.z as f32,
                y_axis.x as f32,
                y_axis.y as f32,
                y_axis.z as f32,
            ],
            shape_type: 3,
            padding: [0; 3],
        }
    }
}

impl Transform for Box3DSdf {
    fn get_model_matrix(&self) -> nalgebra::Matrix4<GMFloat> {
        self.model_matrix
    }
    fn set_model_matrix(&mut self, mat: nalgebra::Matrix4<GMFloat>) {
        self.model_matrix = mat;
    }
}

impl Draw for Box3DSdf {
    fn draw(&self, _ctx: &mut Context, _parent_matrix: nalgebra::Matrix4<GMFloat>) {}
}

impl Mobject for Box3DSdf {
    fn as_3d(&self) -> Option<&dyn crate::mobjects::object_3d::Object3D> {
        Some(self)
    }

    fn set_position(&mut self, pos: nalgebra::Point3<GMFloat>) {
        self.center = pos;
    }

    fn get_position(&self) -> nalgebra::Point3<GMFloat> {
        self.center
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════
