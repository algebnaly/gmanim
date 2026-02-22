pub trait Mobject: Transform + Draw {}
pub trait MobjectClone: Mobject {
    fn mobject_clone(&self) -> Box<dyn MobjectClone>;
}

use std::f32::consts::PI;

use crate::{
    math_utils::k_for_bezier_arc, Color, Context, ContextType, GMFloat, Scene, SceneConfig,
};

use nalgebra::{point, Point, Point2, Point3, Vector2, Vector3};
use tiny_skia::{LineCap, LineJoin, Paint, Stroke, StrokeDash};
pub mod dot;
pub mod formula;
pub mod group;
pub mod path;
pub mod polygon;
pub mod svg_shape;
pub mod text;
pub mod three_d_viewport;
pub use dot::Dot;

pub trait Transform {
    fn transform(&mut self, transform: nalgebra::Transform3<GMFloat>);
    fn scale(&mut self, scale_factor: GMFloat) {
        let scaling_matrix = nalgebra::Matrix4::new_scaling(scale_factor);
        self.transform(nalgebra::Transform::from_matrix_unchecked(scaling_matrix));
    }
    fn move_this(&mut self, movement: nalgebra::Vector3<GMFloat>) {
        let movement_matrix = nalgebra::Matrix4::new_translation(&movement);
        self.transform(nalgebra::Transform::from_matrix_unchecked(movement_matrix));
    }
}

pub trait Draw {
    //draw shape without fill()
    fn draw(&self, ctx: &mut Context);
}

#[derive(Debug, Clone, Copy)]
pub struct DrawConfig {
    stoke_width: GMFloat,
    fill: bool,
    color: Color,
}

impl Default for DrawConfig {
    fn default() -> Self {
        DrawConfig {
            stoke_width: 0.25,
            fill: true,
            color: Default::default(),
        }
    }
}

pub struct Rectangle {
    pub p0: Point3<GMFloat>,
    pub p1: Point3<GMFloat>,
    pub p2: Point3<GMFloat>,
    pub p3: Point3<GMFloat>,
    pub draw_config: DrawConfig,
}

impl Default for Rectangle {
    fn default() -> Self {
        Rectangle {
            p0: Point3::new(0.0, 0.0, 0.0),
            p1: Point3::new(1.0, 0.0, 0.0),
            p2: Point3::new(1.0, 1.0, 0.0),
            p3: Point3::new(0.0, 1.0, 0.0),
            draw_config: DrawConfig::default(),
        }
    }
}

impl Transform for Rectangle {
    fn transform(&mut self, transform: nalgebra::Transform3<GMFloat>) {
        self.p0 = transform * self.p0;
        self.p1 = transform * self.p1;
        self.p2 = transform * self.p2;
        self.p3 = transform * self.p3;
    }
}

impl Draw for Rectangle {
    fn draw(self: &Self, ctx: &mut Context) {
        match &mut ctx.ctx_type {
            ContextType::TinySKIA(pixmap) => {
                let scale_factor = ctx.scene_config.scale_factor;
                let mut pb = tiny_skia::PathBuilder::new();
                let p0 = (
                    coordinate_change_x(self.p0[(0)], ctx.scene_config.width) * scale_factor,
                    coordinate_change_y(self.p0[(1)], ctx.scene_config.height) * scale_factor,
                );
                let p1 = (
                    coordinate_change_x(self.p1[(0)], ctx.scene_config.width) * scale_factor,
                    coordinate_change_y(self.p1[(1)], ctx.scene_config.height) * scale_factor,
                );
                let p2 = (
                    coordinate_change_x(self.p2[(0)], ctx.scene_config.width) * scale_factor,
                    coordinate_change_y(self.p2[(1)], ctx.scene_config.height) * scale_factor,
                );
                let p3 = (
                    coordinate_change_x(self.p3[(0)], ctx.scene_config.width) * scale_factor,
                    coordinate_change_y(self.p3[(1)], ctx.scene_config.height) * scale_factor,
                );
                pb.move_to(p0.0 as f32, p0.1 as f32);
                pb.line_to(p1.0 as f32, p1.1 as f32);
                pb.line_to(p2.0 as f32, p2.1 as f32);
                pb.line_to(p3.0 as f32, p3.1 as f32);
                pb.line_to(p0.0 as f32, p0.1 as f32);
                let path = pb.finish().unwrap();

                let mut stroke = Stroke::default();
                stroke.width = self.draw_config.stoke_width * scale_factor;
                stroke.line_cap = LineCap::Round;
                stroke.line_join = LineJoin::Round;
                let mut paint = Paint::default();
                paint.set_color(self.draw_config.color.into());

                paint.anti_alias = true;
                pixmap.stroke_path(
                    &path,
                    &paint,
                    &stroke,
                    tiny_skia::Transform::identity(),
                    None,
                );
            }
            _ => {}
        }
    }
}

impl Mobject for Rectangle {}

pub struct SimpleLine {
    pub p0: Point3<GMFloat>,
    pub p1: Point3<GMFloat>,
    pub draw_config: DrawConfig,
}

impl Default for SimpleLine {
    fn default() -> Self {
        SimpleLine {
            p0: Point3::new(0.0, 0.0, 0.0),
            p1: Point3::new(1.0, 0.0, 0.0),
            draw_config: DrawConfig::default(),
        }
    }
}

impl Transform for SimpleLine {
    fn transform(&mut self, transform: nalgebra::Transform3<GMFloat>) {
        self.p0 = transform * self.p0;
        self.p1 = transform * self.p1;
    }
}

impl Draw for SimpleLine {
    fn draw(self: &Self, ctx: &mut Context) {
        let scale_factor = ctx.scene_config.scale_factor;
        match &mut ctx.ctx_type {
            ContextType::TinySKIA(pixmap) => {
                let mut pb = tiny_skia::PathBuilder::new();
                let p0 = (
                    coordinate_change_x(self.p0[(0)], ctx.scene_config.width) * scale_factor,
                    coordinate_change_y(self.p0[(1)], ctx.scene_config.height) * scale_factor,
                );
                let p1 = (
                    coordinate_change_x(self.p1[(0)], ctx.scene_config.width) * scale_factor,
                    coordinate_change_y(self.p1[(1)], ctx.scene_config.height) * scale_factor,
                );
                pb.move_to(p0.0 as f32, p0.1 as f32);
                pb.line_to(p1.0 as f32, p1.1 as f32);
                let path = pb.finish().unwrap();

                let mut stroke = Stroke::default();
                stroke.width = self.draw_config.stoke_width * scale_factor;
                stroke.line_cap = LineCap::Round;
                stroke.line_join = LineJoin::Round;
                let mut paint = Paint::default();
                paint.set_color(self.draw_config.color.into());

                pixmap.stroke_path(
                    &path,
                    &paint,
                    &stroke,
                    tiny_skia::Transform::identity(),
                    None,
                );
            }
            _ => {}
        }
    }
}

impl Mobject for SimpleLine {}

pub struct PolyLine {
    pub points: Vec<Point3<GMFloat>>,
    pub draw_config: DrawConfig,
}

impl Default for PolyLine {
    fn default() -> Self {
        PolyLine {
            points: vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 0.0, 0.0)],
            draw_config: DrawConfig::default(),
        }
    }
}

impl Transform for PolyLine {
    fn transform(&mut self, transform: nalgebra::Transform3<GMFloat>) {
        for p in &mut self.points {
            *p = transform * (*p);
        }
    }
}

pub struct Arc {
    center_point: Point3<GMFloat>,
    start_angle: GMFloat,
    end_angle: GMFloat,
    radius: GMFloat,
    _segs: usize,
    _seg_list: Vec<GMFloat>,
    draw_config: DrawConfig,
}

impl Arc {
    pub fn new(
        center_point: Point3<GMFloat>,
        start_angle: GMFloat,
        end_angle: GMFloat,
        radius: GMFloat,
    ) -> Self {
        let _segs = ((end_angle - start_angle) / (PI as GMFloat / 2 as GMFloat)).ceil() as usize;
        let delta_angle = (end_angle - start_angle) / _segs as GMFloat;
        let mut _seg_list = vec![];
        _seg_list.push(start_angle);
        for i in 1..(_segs - 1) {
            _seg_list.push(start_angle + i as GMFloat * delta_angle);
        }
        _seg_list.push(end_angle);
        Self {
            center_point,
            start_angle,
            end_angle,
            radius,
            _segs,
            _seg_list,
            draw_config: DrawConfig::default(),
        }
    }
}

impl Draw for Arc {
    fn draw(&self, ctx: &mut Context) {
        let scale_factor = ctx.scene_config.scale_factor;
        let scene_width = ctx.scene_config.width;
        let scene_height = ctx.scene_config.height;
        match &mut ctx.ctx_type {
            ContextType::TinySKIA(pixmap) => {
                for i in 0..(self._segs - 1) {
                    let mut pb = tiny_skia::PathBuilder::new();
                    // approximate arc by cubic bezier curve here
                    let start_angle = self._seg_list[i];
                    let end_angle = self._seg_list[i + 1];
                    let k = k_for_bezier_arc((end_angle - start_angle) / 2.0);
                    let point_0 = self.center_point.xy()
                        + Vector2::new(end_angle.cos(), end_angle.sin()) * self.radius;

                    let point_3 = self.center_point.xy()
                        + Vector2::new(start_angle.cos(), start_angle.sin()) * self.radius;

                    let point_1 =
                        point_0 + Vector2::new(end_angle.sin(), -end_angle.cos()) * k * self.radius;
                    let point_2 = point_3
                        + Vector2::new(-start_angle.sin(), start_angle.cos()) * k * self.radius;
                    pb.move_to(
                        coordinate_change_x(point_0.x, scene_width) * scale_factor,
                        coordinate_change_y(point_0.y, scene_height) * scale_factor,
                    );
                    pb.cubic_to(
                        coordinate_change_x(point_1.x, scene_width) * scale_factor,
                        coordinate_change_y(point_1.y, scene_height) * scale_factor,
                        coordinate_change_x(point_2.x, scene_width) * scale_factor,
                        coordinate_change_y(point_2.y, scene_height) * scale_factor,
                        coordinate_change_x(point_3.x, scene_width) * scale_factor,
                        coordinate_change_y(point_3.y, scene_height) * scale_factor,
                    );

                    let path = pb.finish().unwrap();
                    let mut stroke = Stroke::default();
                    stroke.width = self.draw_config.stoke_width * scale_factor;
                    stroke.line_cap = LineCap::Round;
                    stroke.line_join = LineJoin::Round;
                    let mut paint = Paint::default();
                    paint.set_color(self.draw_config.color.into());

                    pixmap.stroke_path(
                        &path,
                        &paint,
                        &stroke,
                        tiny_skia::Transform::identity(),
                        None,
                    );
                }
            }
            _ => {}
        }
    }
}

impl Transform for Arc {
    fn transform(&mut self, transform: nalgebra::Transform3<GMFloat>) {}
}

impl Mobject for Arc {}

impl Draw for PolyLine {
    fn draw(self: &Self, ctx: &mut Context) {
        if self.points.len() < 2 {
            return;
        }

        let scale_factor = ctx.scene_config.scale_factor;

        match &mut ctx.ctx_type {
            ContextType::TinySKIA(pixmap) => {
                let mut pb = tiny_skia::PathBuilder::new();
                let p0 = (
                    coordinate_change_x(self.points[0][(0)], ctx.scene_config.width) * scale_factor,
                    coordinate_change_y(self.points[0][(1)], ctx.scene_config.height)
                        * scale_factor,
                );
                pb.move_to(p0.0 as f32, p0.1 as f32);
                for p in self.points[1..].iter() {
                    let point = (
                        coordinate_change_x(p[(0)], ctx.scene_config.width) * scale_factor,
                        coordinate_change_y(p[(1)], ctx.scene_config.height) * scale_factor,
                    );
                    pb.line_to(point.0 as f32, point.1 as f32);
                }
                let path = pb.finish().unwrap();

                let mut stroke = Stroke::default();
                stroke.width = self.draw_config.stoke_width * scale_factor;
                stroke.line_cap = LineCap::Round;
                stroke.line_join = LineJoin::Round;

                let mut paint = Paint::default();
                paint.set_color(self.draw_config.color.into());

                pixmap.stroke_path(
                    &path,
                    &paint,
                    &stroke,
                    tiny_skia::Transform::identity(),
                    None,
                );
            }
            _ => {}
        }
    }
}

impl Mobject for PolyLine {}

pub fn rotate_matrix(axis: Vector3<GMFloat>, theta: GMFloat) {
    //assume axis is a unit vector
}

#[inline]
pub fn coordinate_change_x(position_x: GMFloat, scene_width: GMFloat) -> GMFloat {
    scene_width / 2.0 + position_x
}

#[inline]
pub fn coordinate_change_y(position_y: GMFloat, scene_height: GMFloat) -> GMFloat {
    scene_height / 2.0 - position_y
}

#[test]
fn test_draw_arc() {
    let mut ctx = Context::default();
    let mut scene = Scene::default();
    let arc = Arc::new(Point3::new(0.0, 1.0, 0.0), 0.0, PI as GMFloat * 2.0, 3.0);
    scene.add(Box::new(arc));
    scene.save_png(&mut ctx, "arc.png");
}
