use nalgebra::{Point3, Vector3};
use rayon::prelude::*;
use std::sync::Arc;

use crate::camera::Camera;
use crate::mobjects::mesh_3d::SurfaceMaterial;
use crate::mobjects::{Draw, Mobject};
use crate::{Context, GMFloat};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SdfPrimitive {
    Sphere {
        center: Point3<GMFloat>,
        radius: GMFloat,
    },
    Capsule {
        start: Point3<GMFloat>,
        end: Point3<GMFloat>,
        radius: GMFloat,
    },
    Arrow {
        start: Point3<GMFloat>,
        end: Point3<GMFloat>,
        shaft_radius: GMFloat,
        head_radius: GMFloat,
        head_length: GMFloat,
    },
    OrientedBox {
        center: Point3<GMFloat>,
        half_extents: Vector3<GMFloat>,
        x_axis: Vector3<GMFloat>,
        y_axis: Vector3<GMFloat>,
    },
    QuadraticBezier {
        start: Point3<GMFloat>,
        control: Point3<GMFloat>,
        end: Point3<GMFloat>,
        radius: GMFloat,
    },
}

/// Represents an object defined by a Signed Distance Function.
pub trait Object3D: Send + Sync {
    /// Returns the shortest distance from point `p` to the surface of the object (Used by CPU rendering).
    fn distance(&self, p: &Point3<GMFloat>) -> GMFloat;
    fn material(&self) -> SurfaceMaterial;

    fn transformed_primitive(&self, global_mat: nalgebra::Matrix4<GMFloat>) -> SdfPrimitive;
}

// ═══════════════════════════════════════════════════════════════════════════
// Primitive 3D Objects
// ═══════════════════════════════════════════════════════════════════════════

/// A 3D Sphere
pub struct Sphere3D {
    pub radius: GMFloat,
    pub material: SurfaceMaterial,
}

impl Object3D for Sphere3D {
    fn distance(&self, p: &Point3<GMFloat>) -> GMFloat {
        (p - Point3::origin()).norm() - self.radius
    }
    fn material(&self) -> SurfaceMaterial {
        self.material
    }
    fn transformed_primitive(&self, global_mat: nalgebra::Matrix4<GMFloat>) -> SdfPrimitive {
        let center = global_mat.transform_point(&Point3::origin());
        SdfPrimitive::Sphere {
            center,
            radius: self.radius,
        }
    }
}
impl Draw for Sphere3D {
    fn draw(&self, _ctx: &mut Context, _parent_matrix: nalgebra::Matrix4<GMFloat>) {}
}
impl Mobject for Sphere3D {
    fn default_name(&self) -> &'static str {
        "Sphere3D"
    }

    fn submit_to_renderer(
        &self,
        visitor: &mut dyn crate::mobjects::RenderVisitor,
        world_transform: nalgebra::Matrix4<crate::GMFloat>,
    ) {
        visitor.push_surface_3d(crate::mobjects::Surface3DSubmission {
            geometry: crate::mobjects::Geometry3DRef::Sdf(self),
            material: self.material,
            transform: world_transform,
        });
    }
}

/// A 3D Line Segment (Capsule)
pub struct LineSegment3D {
    pub a: Point3<GMFloat>,
    pub b: Point3<GMFloat>,
    pub radius: GMFloat,
    pub material: SurfaceMaterial,
}

impl Object3D for LineSegment3D {
    fn distance(&self, p: &Point3<GMFloat>) -> GMFloat {
        let pa = p - self.a;
        let ba = self.b - self.a;
        let h = (pa.dot(&ba) / ba.dot(&ba)).clamp(0.0, 1.0);
        (pa - ba * h).norm() - self.radius
    }
    fn material(&self) -> SurfaceMaterial {
        self.material
    }
    fn transformed_primitive(&self, global_mat: nalgebra::Matrix4<GMFloat>) -> SdfPrimitive {
        SdfPrimitive::Capsule {
            start: global_mat.transform_point(&self.a),
            end: global_mat.transform_point(&self.b),
            radius: self.radius,
        }
    }
}
impl Draw for LineSegment3D {
    fn draw(&self, _ctx: &mut Context, _parent_matrix: nalgebra::Matrix4<GMFloat>) {}
}
impl Mobject for LineSegment3D {
    fn default_name(&self) -> &'static str {
        "LineSegment3D"
    }
    fn submit_to_renderer(
        &self,
        visitor: &mut dyn crate::mobjects::RenderVisitor,
        world_transform: nalgebra::Matrix4<crate::GMFloat>,
    ) {
        visitor.push_surface_3d(crate::mobjects::Surface3DSubmission {
            geometry: crate::mobjects::Geometry3DRef::Sdf(self),
            material: self.material,
            transform: world_transform,
        });
    }
}

/// A 3D Quadratic Bezier Curve
pub struct QuadraticBezier3D {
    pub a: Point3<GMFloat>,
    pub b: Point3<GMFloat>,
    pub c: Point3<GMFloat>,
    pub radius: GMFloat,
    pub material: SurfaceMaterial,
}

impl Object3D for QuadraticBezier3D {
    fn distance(&self, p: &Point3<GMFloat>) -> GMFloat {
        let pa = p - self.a;
        let ca = self.c - self.a;
        let h = (pa.dot(&ca) / ca.dot(&ca)).clamp(0.0, 1.0);
        (pa - ca * h).norm() - self.radius
    }
    fn material(&self) -> SurfaceMaterial {
        self.material
    }
    fn transformed_primitive(&self, global_mat: nalgebra::Matrix4<GMFloat>) -> SdfPrimitive {
        SdfPrimitive::QuadraticBezier {
            start: global_mat.transform_point(&self.a),
            control: global_mat.transform_point(&self.b),
            end: global_mat.transform_point(&self.c),
            radius: self.radius,
        }
    }
}
impl Draw for QuadraticBezier3D {
    fn draw(&self, _ctx: &mut Context, _parent_matrix: nalgebra::Matrix4<GMFloat>) {}
}
impl Mobject for QuadraticBezier3D {
    fn default_name(&self) -> &'static str {
        "QuadraticBezier3D"
    }

    fn submit_to_renderer(
        &self,
        visitor: &mut dyn crate::mobjects::RenderVisitor,
        world_transform: nalgebra::Matrix4<crate::GMFloat>,
    ) {
        visitor.push_surface_3d(crate::mobjects::Surface3DSubmission {
            geometry: crate::mobjects::Geometry3DRef::Sdf(self),
            material: self.material,
            transform: world_transform,
        });
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
    pub material: SurfaceMaterial,
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

    fn material(&self) -> SurfaceMaterial {
        self.material
    }

    fn transformed_primitive(&self, global_mat: nalgebra::Matrix4<GMFloat>) -> SdfPrimitive {
        SdfPrimitive::Arrow {
            start: global_mat.transform_point(&self.start),
            end: global_mat.transform_point(&self.end),
            shaft_radius: self.shaft_radius,
            head_radius: self.head_radius,
            head_length: self.head_length,
        }
    }
}
impl Draw for Arrow3D {
    fn draw(&self, _ctx: &mut Context, _parent_matrix: nalgebra::Matrix4<GMFloat>) {}
}
impl Mobject for Arrow3D {
    fn default_name(&self) -> &'static str {
        "Arrow3D"
    }

    fn submit_to_renderer(
        &self,
        visitor: &mut dyn crate::mobjects::RenderVisitor,
        world_transform: nalgebra::Matrix4<crate::GMFloat>,
    ) {
        visitor.push_surface_3d(crate::mobjects::Surface3DSubmission {
            geometry: crate::mobjects::Geometry3DRef::Sdf(self),
            material: self.material,
            transform: world_transform,
        });
    }
}

pub struct Box3DSdf {
    pub size: Vector3<GMFloat>,
    pub x_axis: Vector3<GMFloat>,
    pub y_axis: Vector3<GMFloat>,
    pub z_axis: Vector3<GMFloat>,
    pub material: SurfaceMaterial,
}

impl Object3D for Box3DSdf {
    fn distance(&self, p: &Point3<GMFloat>) -> GMFloat {
        let pt = p - Point3::origin();
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

    fn material(&self) -> SurfaceMaterial {
        self.material
    }

    fn transformed_primitive(&self, global_mat: nalgebra::Matrix4<GMFloat>) -> SdfPrimitive {
        let center = global_mat.transform_point(&Point3::origin());
        let x_axis = global_mat.transform_vector(&self.x_axis);
        let y_axis = global_mat.transform_vector(&self.y_axis);
        SdfPrimitive::OrientedBox {
            center,
            half_extents: self.size,
            x_axis,
            y_axis,
        }
    }
}

impl Draw for Box3DSdf {
    fn draw(&self, _ctx: &mut Context, _parent_matrix: nalgebra::Matrix4<GMFloat>) {}
}

impl Mobject for Box3DSdf {
    fn default_name(&self) -> &'static str {
        "Box3DSdf"
    }

    fn submit_to_renderer(
        &self,
        visitor: &mut dyn crate::mobjects::RenderVisitor,
        world_transform: nalgebra::Matrix4<crate::GMFloat>,
    ) {
        visitor.push_surface_3d(crate::mobjects::Surface3DSubmission {
            geometry: crate::mobjects::Geometry3DRef::Sdf(self),
            material: self.material,
            transform: world_transform,
        });
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════
