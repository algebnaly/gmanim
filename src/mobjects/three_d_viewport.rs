use std::f32::INFINITY;

use crate::{math_utils::constants::PI, mobjects::Transform, Color};
use nalgebra::{Isometry2, Matrix2, Point2, Point3, Point4, RealField};

use crate::{camera::Camera, GMFloat};

use super::Draw;

struct ThreeDViewport {
    pub position: Point3<GMFloat>,
    pub vp_width: GMFloat,
    pub vp_height: GMFloat,
    pub camera: Camera,
    pub triangle_list: Vec<Triangle>,
    pub model_matrix: nalgebra::Matrix4<GMFloat>,
}
struct Triangle {
    p0: Point3<GMFloat>,
    p1: Point3<GMFloat>,
    p2: Point3<GMFloat>,
}

impl ThreeDViewport {
    pub fn new(
        position: Point3<GMFloat>,
        vp_width: GMFloat,
        vp_height: GMFloat,
        camera: Camera,
    ) -> Self {
        Self {
            position,
            vp_width,
            vp_height,
            camera,
            triangle_list: Vec::new(),
            model_matrix: nalgebra::Matrix4::identity(),
        }
    }
}

impl Default for ThreeDViewport {
    fn default() -> Self {
        Self {
            position: Point3::origin(),
            vp_width: 16.0,
            vp_height: 9.0,
            camera: Camera::default(),
            triangle_list: Vec::new(),
            model_matrix: nalgebra::Matrix4::identity(),
        }
    }
}

impl Transform for ThreeDViewport {
    fn get_model_matrix(&self) -> nalgebra::Matrix4<GMFloat> {
        self.model_matrix
    }
    fn set_model_matrix(&mut self, mat: nalgebra::Matrix4<GMFloat>) {
        self.model_matrix = mat;
    }
}

impl Draw for ThreeDViewport {
    fn draw(&self, _ctx: &mut crate::Context, _parent_matrix: nalgebra::Matrix4<crate::GMFloat>) {
        // Will be replaced by GPU logic
    }
}

#[test]
pub fn test_three_d() {
    // Test is skipped due to removing tiny-skia
}

#[inline]
pub fn try_triangle_inner_z(
    p0: Point3<GMFloat>,
    p1: Point3<GMFloat>,
    p2: Point3<GMFloat>,
    p: Point2<GMFloat>,
) -> Option<GMFloat> {
    // test if p is in triangle and give the z value
    let mut rotate = Isometry2::rotation(PI / 2.0);
    let v0 = p1 - p0;
    let v1 = p2 - p1;
    let v2 = p0 - p2;
    if (rotate * v0.xy()).dot(&v1.xy()) < 0.0 {
        rotate = rotate.inverse();
    }

    let d0 = rotate * v0.xy();
    let d1 = rotate * v1.xy();
    let d2 = rotate * v2.xy();
    let n_v0 = p - p0.xy();
    let n_v1 = p - p1.xy();
    let n_v2 = p - p2.xy();

    if !(d0.dot(&n_v0).is_sign_positive()
        && d1.dot(&n_v1).is_sign_positive()
        && d2.dot(&n_v2).is_sign_positive())
    {
        return None;
    }

    let basis_matrix = Matrix2::from_columns(&[v0.xy(), v1.xy()]);
    let maybe_b_inv = basis_matrix.try_inverse();
    if maybe_b_inv.is_none() {
        return None;
    }
    let b_inv = maybe_b_inv.unwrap();
    let c = b_inv * p;
    let z = c[0] * v0[2] + c[1] * v1[2];
    Some(z)
}
