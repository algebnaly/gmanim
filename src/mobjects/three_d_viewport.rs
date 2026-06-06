use std::f32::INFINITY;

use crate::{math_utils::constants::PI, mobjects::Transform, Color};
use nalgebra::{Isometry2, Matrix2, Point2, Point3, Point4, RealField};
use tiny_skia::{Pixmap, PixmapPaint};

use crate::{camera::Camera, GMFloat};

use super::Draw;

struct ThreeDViewport {
    pub position: Point3<GMFloat>,
    pub vp_width: GMFloat,
    pub vp_height: GMFloat,
    pub camera: Camera,
    pub triangle_list: Vec<Triangle>,
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
        }
    }
}

impl Transform for ThreeDViewport {
    fn transform(&mut self, transform: nalgebra::Transform3<GMFloat>) {
        self.position = transform * self.position;
    }
}

impl Draw for ThreeDViewport {
    fn draw(&self, ctx: &mut crate::Context) {
        let pixmap_size = (
            (self.vp_width * ctx.scene_config.scale_factor) as u32,
            (self.vp_height * ctx.scene_config.scale_factor) as u32,
        );
        let target_pix_coord = (
            ctx.scene_config.convert_coord_y(self.position.x) as i32,
            ctx.scene_config.convert_coord_y(self.position.y) as i32,
        );
        if pixmap_size.0 == 0 {
            return;
        }
        if pixmap_size.1 == 0 {
            return;
        }
        let mut z_buffer: Vec<Vec<GMFloat>> = (0..pixmap_size.1)
            .map(|_| (0..pixmap_size.0).map(|_| -GMFloat::INFINITY).collect())
            .collect();
        let mut new_pixmap = Pixmap::new(pixmap_size.0, pixmap_size.1).unwrap();
        for t in &self.triangle_list {
            let camera_transform = self.camera.get_camera_transform_matrix();
            let projection_transform = self.camera.get_projection_transform_matrix();
            let m = projection_transform * camera_transform;
            let p0_p = m * t.p0.to_homogeneous();
            let p1_p = m * t.p1.to_homogeneous();
            let p2_p = m * t.p2.to_homogeneous();
            let p_x_list = [
                (p0_p[0] + 1.0) / 2.0 * pixmap_size.0 as GMFloat,
                (p1_p[0] + 1.0) / 2.0 * pixmap_size.0 as GMFloat,
                (p2_p[0] + 1.0) / 2.0 * pixmap_size.0 as GMFloat,
            ];
            let p_y_list = [
                (p0_p[1] + 1.0) / 2.0 * pixmap_size.1 as GMFloat,
                (p1_p[1] + 1.0) / 2.0 * pixmap_size.1 as GMFloat,
                (p2_p[1] + 1.0) / 2.0 * pixmap_size.1 as GMFloat,
            ];

            let x_min = p_x_list
                .into_iter()
                .map(|x| x as i32)
                .min()
                .unwrap()
                .clone()
                .clamp(0, pixmap_size.0 as i32);
            let x_max = p_x_list
                .into_iter()
                .map(|x| x as i32)
                .max()
                .unwrap()
                .clone()
                .clamp(0, pixmap_size.0 as i32);
            let y_min = p_y_list
                .into_iter()
                .map(|y| y as i32)
                .min()
                .unwrap()
                .clone()
                .clamp(0, pixmap_size.1 as i32);
            let y_max = p_y_list
                .into_iter()
                .map(|y| y as i32)
                .max()
                .unwrap()
                .clone()
                .clamp(0, pixmap_size.1 as i32);
            for x in x_min..x_max {
                for y in y_min..y_max {
                    let half_width = pixmap_size.0 as GMFloat / 2.0;
                    let half_height = pixmap_size.1 as GMFloat / 2.0;
                    let x_r = x as GMFloat / half_width - 1.0;
                    let y_r = y as GMFloat / half_height - 1.0;

                    if let Some(new_z) = try_triangle_inner_z(
                        Point3::from_homogeneous(p0_p).unwrap(),
                        Point3::from_homogeneous(p1_p).unwrap(),
                        Point3::from_homogeneous(p2_p).unwrap(),
                        Point2::new(x_r, y_r),
                    ) {
                        if z_buffer[y as usize][x as usize] < new_z {
                            z_buffer[y as usize][x as usize] = new_z;
                            let pix_list = new_pixmap.pixels_mut();
                            pix_list[y as usize * pixmap_size.0 as usize + x as usize] =
                                tiny_skia::PremultipliedColorU8::from_rgba(255, 255, 0, 255)
                                    .unwrap();
                        }
                    }
                }
            }
        }
        new_pixmap.save_png("out1.png");
        ctx.pixmap.draw_pixmap(
            0,
            0,
            new_pixmap.as_ref(),
            &PixmapPaint::default(),
            tiny_skia::Transform::identity(),
            None,
        );
    }
}

#[test]
pub fn test_three_d() {
    let mut ctx = crate::Context::default();
    let mut scene = crate::Scene::new();
    let mut three_d_vp = ThreeDViewport::default();
    three_d_vp.camera.position = Point3::new(0.0, 0.0, 6.0);
    three_d_vp.triangle_list.push(Triangle {
        p0: Point3::new(0.0, 0.0, 0.0),
        p1: Point3::new(0.5, 0.25, 0.0),
        p2: Point3::new(0.75, 0.66, 0.66),
    });
    three_d_vp.draw(&mut ctx);
    ctx.pixmap.save_png("output.png").unwrap();
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
