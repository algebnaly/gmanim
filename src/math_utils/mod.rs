use nalgebra::Vector2;

use crate::{
    Color, Context, GMFloat,
    mobjects::{coordinate_change_x, coordinate_change_y},
};

pub mod constants;

pub fn bezier_curve(point_list: &[Vector2<GMFloat>], t: GMFloat) -> Vector2<GMFloat> {
    if point_list.len() < 2 {
        panic!("bezier curve with fewer than two point");
    }
    if point_list.len() == 2 {
        return lerp(point_list[0], point_list[1], t);
    }
    let mut new_point_list = vec![];
    for i in 0..point_list.len() - 1 {
        new_point_list.push(lerp(point_list[i], point_list[i + 1], t));
    }
    return bezier_curve(&new_point_list, t);
}

// TODO: use bernstein polynominal to speed up computation
pub fn bezier_cubic(
    p0: Vector2<GMFloat>,
    p1: Vector2<GMFloat>,
    p2: Vector2<GMFloat>,
    t: GMFloat,
) -> Vector2<GMFloat> {
    let p0_1 = lerp(p0, p1, t);
    let p1_1 = lerp(p1, p2, t);
    let p2 = lerp(p0_1, p1_1, t);
    return p2;
}

pub fn bezier_quad(
    p0: Vector2<GMFloat>,
    p1: Vector2<GMFloat>,
    p2: Vector2<GMFloat>,
    p3: Vector2<GMFloat>,
    t: GMFloat,
) -> Vector2<GMFloat> {
    let p0_1 = lerp(p0, p1, t);
    let p1_1 = lerp(p1, p2, t);
    let p2_1 = lerp(p2, p3, t);
    let p3 = bezier_cubic(p0_1, p1_1, p2_1, t);
    return p3;
}

pub fn bezier_5(
    p0: Vector2<GMFloat>,
    p1: Vector2<GMFloat>,
    p2: Vector2<GMFloat>,
    p3: Vector2<GMFloat>,
    p4: Vector2<GMFloat>,
    t: GMFloat,
) -> Vector2<GMFloat> {
    let p0_1 = lerp(p0, p1, t);
    let p1_1 = lerp(p1, p2, t);
    let p2_1 = lerp(p2, p3, t);
    let p3_1 = lerp(p3, p4, t);
    let p4 = bezier_quad(p0_1, p1_1, p2_1, p3_1, t);
    return p4;
}

pub fn bezier_6(
    p0: Vector2<GMFloat>,
    p1: Vector2<GMFloat>,
    p2: Vector2<GMFloat>,
    p3: Vector2<GMFloat>,
    p4: Vector2<GMFloat>,
    p5: Vector2<GMFloat>,
    t: GMFloat,
) -> Vector2<GMFloat> {
    let p0_1 = lerp(p0, p1, t);
    let p1_1 = lerp(p1, p2, t);
    let p2_1 = lerp(p2, p3, t);
    let p3_1 = lerp(p3, p4, t);
    let p4_1 = lerp(p4, p5, t);
    let p5 = bezier_5(p0_1, p1_1, p2_1, p3_1, p4_1, t);
    return p5;
}

#[inline]
pub fn lerp(
    p0: nalgebra::Vector2<GMFloat>,
    p1: nalgebra::Vector2<GMFloat>,
    t: GMFloat,
) -> nalgebra::Vector2<GMFloat> {
    return p0 * (1.0 - t) + p1 * t;
}

pub fn k_for_bezier_arc(theta: GMFloat) -> GMFloat {
    // assume theta was less than pi/2
    (4.0 / 3.0 * (1.0 - (theta as GMFloat).cos()) / (theta as GMFloat).sin()) as GMFloat
}

#[inline]
pub fn point2d_to_point3d(p: nalgebra::Point2<GMFloat>) -> nalgebra::Point3<GMFloat> {
    nalgebra::Point3::new(p.x, p.y, 0.0)
}

#[inline]
pub fn point3d_to_point2d(p: nalgebra::Point3<GMFloat>) -> nalgebra::Point2<GMFloat> {
    nalgebra::Point2::new(p.x, p.y)
}
