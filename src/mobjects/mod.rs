pub trait Mobject: Transform + Draw {
    fn get_position(&self) -> nalgebra::Point3<GMFloat> {
        let mat = self.get_model_matrix();
        nalgebra::Point3::new(mat.m14, mat.m24, mat.m34)
    }
    fn set_position(&mut self, pos: nalgebra::Point3<GMFloat>) {
        let mut mat = self.get_model_matrix();
        mat.m14 = pos.x;
        mat.m24 = pos.y;
        mat.m34 = pos.z;
        self.set_model_matrix(mat);
    }
    fn as_3d(&self) -> Option<&dyn crate::mobjects::object_3d::Object3D> {
        None
    }
}
pub trait MobjectClone: Mobject {
    fn mobject_clone(&self) -> Box<dyn MobjectClone>;
}

use std::f32::consts::PI;

use crate::{math_utils::k_for_bezier_arc, Color, Context, GMFloat, Scene, SceneConfig};

use nalgebra::{point, Point, Point2, Point3, Vector2, Vector3};
use tiny_skia::{LineCap, LineJoin, Paint, Stroke, StrokeDash};
pub mod dot;
pub mod formula;
pub mod group;
pub mod object_3d;
pub mod path;
pub mod polygon;
pub mod svg_shape;
pub mod text;
pub mod three_d_viewport;
pub use dot::Dot;

pub trait Transform {
    // Modify the model_matrix natively
    fn get_model_matrix(&self) -> nalgebra::Matrix4<GMFloat>;
    fn set_model_matrix(&mut self, mat: nalgebra::Matrix4<GMFloat>);

    fn apply_transform(&mut self, transform: nalgebra::Matrix4<GMFloat>) {
        let current = self.get_model_matrix();
        self.set_model_matrix(transform * current);
    }
    
    fn move_this(&mut self, movement: nalgebra::Vector3<GMFloat>) {
        let movement_matrix = nalgebra::Matrix4::new_translation(&movement);
        self.apply_transform(movement_matrix);
    }
    
    fn scale(&mut self, scale_factor: GMFloat) {
        let scaling_matrix = nalgebra::Matrix4::new_scaling(scale_factor);
        self.apply_transform(scaling_matrix);
    }
}

pub trait Draw {
    // draw shape, incorporating accumulated parent transformations
    fn draw(&self, ctx: &mut Context, parent_matrix: nalgebra::Matrix4<GMFloat>);
}

#[derive(Debug, Clone, Copy)]
pub struct DrawConfig {
    pub stoke_width: GMFloat,
    pub fill: bool,
    pub color: Color,
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

impl DrawConfig {
    pub fn get_stroke(&self, scale_factor: f32) -> Option<tiny_skia::Stroke> {
        if self.stoke_width <= 0.0 {
            return None;
        }
        let mut stroke = tiny_skia::Stroke::default();
        stroke.width = self.stoke_width as f32 * scale_factor;
        stroke.line_cap = tiny_skia::LineCap::Round;
        stroke.line_join = tiny_skia::LineJoin::Round;
        Some(stroke)
    }
}

pub struct Rectangle {
    pub p0: Point3<GMFloat>, // Top left
    pub p1: Point3<GMFloat>, // Bottom left
    pub p2: Point3<GMFloat>, // Bottom right
    pub p3: Point3<GMFloat>, // Top right
    pub color: Color,
    pub draw_config: DrawConfig,
    pub model_matrix: nalgebra::Matrix4<GMFloat>,
}

impl Default for Rectangle {
    fn default() -> Self {
        Rectangle {
            p0: Point3::new(0.0, 0.0, 0.0),
            p1: Point3::new(1.0, 0.0, 0.0),
            p2: Point3::new(1.0, 1.0, 0.0),
            p3: Point3::new(0.0, 1.0, 0.0),
            color: Color::default(),
            draw_config: DrawConfig::default(),
            model_matrix: nalgebra::Matrix4::identity(),
        }
    }
}

impl Transform for Rectangle {
    fn get_model_matrix(&self) -> nalgebra::Matrix4<GMFloat> {
        self.model_matrix
    }
    fn set_model_matrix(&mut self, mat: nalgebra::Matrix4<GMFloat>) {
        self.model_matrix = mat;
    }
}

impl Draw for Rectangle {
    fn draw(&self, ctx: &mut Context, parent_matrix: nalgebra::Matrix4<GMFloat>) {
        let global_mat = parent_matrix * self.model_matrix;
        let ts_transform = tiny_skia::Transform::from_row(
            global_mat.m11 as f32, global_mat.m21 as f32,
            global_mat.m12 as f32, global_mat.m22 as f32,
            global_mat.m14 as f32, global_mat.m24 as f32,
        );

        let mut pb = tiny_skia::PathBuilder::new();
        pb.move_to(
            ctx.scene_config.convert_coord_x(self.p0.x),
            ctx.scene_config.convert_coord_y(self.p0.y),
        );
        pb.line_to(
            ctx.scene_config.convert_coord_x(self.p1.x),
            ctx.scene_config.convert_coord_y(self.p1.y),
        );
        pb.line_to(
            ctx.scene_config.convert_coord_x(self.p2.x),
            ctx.scene_config.convert_coord_y(self.p2.y),
        );
        pb.line_to(
            ctx.scene_config.convert_coord_x(self.p3.x),
            ctx.scene_config.convert_coord_y(self.p3.y),
        );
        pb.close();
        let path = pb.finish().unwrap();

        let mut paint = tiny_skia::Paint::default();
        paint.set_color(self.color.into());

        if let Some(mut st) = self.draw_config.get_stroke(ctx.scene_config.scale_factor) {
            ctx.pixmap.stroke_path(
                &path,
                &paint,
                &st,
                ts_transform,
                None,
            );
        }

        if self.draw_config.fill {
            ctx.pixmap.fill_path(
                &path,
                &paint,
                tiny_skia::FillRule::EvenOdd,
                ts_transform,
                None,
            );
        }
    }
}

impl Mobject for Rectangle {}

pub struct SimpleLine {
    pub p0: Point3<GMFloat>,
    pub p1: Point3<GMFloat>,
    pub draw_config: DrawConfig,
    pub model_matrix: nalgebra::Matrix4<GMFloat>,
}

impl SimpleLine {
    pub fn new(p0: Point3<GMFloat>, p1: Point3<GMFloat>) -> Self {
        Self {
            p0,
            p1,
            draw_config: DrawConfig::default(),
            model_matrix: nalgebra::Matrix4::identity(),
        }
    }
}

impl Transform for SimpleLine {
    fn get_model_matrix(&self) -> nalgebra::Matrix4<GMFloat> {
        self.model_matrix
    }
    fn set_model_matrix(&mut self, mat: nalgebra::Matrix4<GMFloat>) {
        self.model_matrix = mat;
    }
}

impl Draw for SimpleLine {
    fn draw(&self, ctx: &mut Context, parent_matrix: nalgebra::Matrix4<GMFloat>) {
        let global_mat = parent_matrix * self.model_matrix;
        let ts_transform = tiny_skia::Transform::from_row(
            global_mat.m11 as f32, global_mat.m21 as f32,
            global_mat.m12 as f32, global_mat.m22 as f32,
            global_mat.m14 as f32, global_mat.m24 as f32,
        );

        let mut pb = tiny_skia::PathBuilder::new();
        pb.move_to(
            ctx.scene_config.convert_coord_x(self.p0.x),
            ctx.scene_config.convert_coord_y(self.p0.y),
        );
        pb.line_to(
            ctx.scene_config.convert_coord_x(self.p1.x),
            ctx.scene_config.convert_coord_y(self.p1.y),
        );
        if let Some(path) = pb.finish() {
            let mut paint = tiny_skia::Paint::default();
            paint.set_color(self.draw_config.color.into());

            if let Some(mut stroke) = self.draw_config.get_stroke(ctx.scene_config.scale_factor) {
                ctx.pixmap.stroke_path(
                    &path,
                    &paint,
                    &stroke,
                    ts_transform,
                    None,
                );
            }
        }
    }
}

impl Mobject for SimpleLine {}

pub struct PolyLine {
    pub points: Vec<Point3<GMFloat>>,
    pub draw_config: DrawConfig,
    pub model_matrix: nalgebra::Matrix4<GMFloat>,
}

impl PolyLine {
    pub fn new(points: Vec<Point3<GMFloat>>) -> Self {
        Self {
            points,
            draw_config: DrawConfig::default(),
            model_matrix: nalgebra::Matrix4::identity(),
        }
    }
}

impl Transform for PolyLine {
    fn get_model_matrix(&self) -> nalgebra::Matrix4<GMFloat> {
        self.model_matrix
    }
    fn set_model_matrix(&mut self, mat: nalgebra::Matrix4<GMFloat>) {
        self.model_matrix = mat;
    }
}

pub struct Arc {
    pub center_point: Point3<GMFloat>,
    pub start_angle: GMFloat,
    pub end_angle: GMFloat,
    pub radius: GMFloat,
    pub draw_config: DrawConfig,
    pub model_matrix: nalgebra::Matrix4<GMFloat>,
}

impl Arc {
    pub fn new(
        center_point: Point3<GMFloat>,
        start_angle: GMFloat,
        end_angle: GMFloat,
        radius: GMFloat,
    ) -> Self {
        Self {
            center_point,
            start_angle,
            end_angle,
            radius,
            draw_config: DrawConfig::default(),
            model_matrix: nalgebra::Matrix4::identity(),
        }
    }
}

impl Draw for Arc {
    fn draw(&self, ctx: &mut Context, parent_matrix: nalgebra::Matrix4<GMFloat>) {
        let global_mat = parent_matrix * self.model_matrix;
        let ts_transform = tiny_skia::Transform::from_row(
            global_mat.m11 as f32, global_mat.m21 as f32,
            global_mat.m12 as f32, global_mat.m22 as f32,
            global_mat.m14 as f32, global_mat.m24 as f32,
        );

        let mut pb = tiny_skia::PathBuilder::new();
        
        let num_curves = ((self.end_angle - self.start_angle).abs() / (PI / 2.0)).ceil() as usize;
        if num_curves == 0 { return; }

        let angle_step = (self.end_angle - self.start_angle) / (num_curves as f32);
        
        let start_x = self.center_point.x + self.radius * self.start_angle.cos();
        let start_y = self.center_point.y + self.radius * self.start_angle.sin();
        pb.move_to(
            ctx.scene_config.convert_coord_x(start_x),
            ctx.scene_config.convert_coord_y(start_y)
        );

        let mut current_angle = self.start_angle;
        for _ in 0..num_curves {
            let next_angle = current_angle + angle_step;
            let k = k_for_bezier_arc(angle_step);
            
            let cp1_x = self.center_point.x + self.radius * (current_angle.cos() - k * current_angle.sin());
            let cp1_y = self.center_point.y + self.radius * (current_angle.sin() + k * current_angle.cos());
            
            let cp2_x = self.center_point.x + self.radius * (next_angle.cos() + k * next_angle.sin());
            let cp2_y = self.center_point.y + self.radius * (next_angle.sin() - k * current_angle.cos());
            
            let end_x = self.center_point.x + self.radius * next_angle.cos();
            let end_y = self.center_point.y + self.radius * next_angle.sin();

            pb.cubic_to(
                ctx.scene_config.convert_coord_x(cp1_x), ctx.scene_config.convert_coord_y(cp1_y),
                ctx.scene_config.convert_coord_x(cp2_x), ctx.scene_config.convert_coord_y(cp2_y),
                ctx.scene_config.convert_coord_x(end_x), ctx.scene_config.convert_coord_y(end_y)
            );
            current_angle = next_angle;
        }

        if let Some(path) = pb.finish() {
            let mut paint = tiny_skia::Paint::default();
            paint.set_color(self.draw_config.color.into());

            if let Some(mut stroke) = self.draw_config.get_stroke(ctx.scene_config.scale_factor) {
                ctx.pixmap.stroke_path(
                    &path,
                    &paint,
                    &stroke,
                    ts_transform,
                    None,
                );
            }
        }
    }
}

impl Transform for Arc {
    fn get_model_matrix(&self) -> nalgebra::Matrix4<GMFloat> {
        self.model_matrix
    }
    fn set_model_matrix(&mut self, mat: nalgebra::Matrix4<GMFloat>) {
        self.model_matrix = mat;
    }
}

impl Mobject for Arc {}

impl Draw for PolyLine {
    fn draw(&self, ctx: &mut Context, parent_matrix: nalgebra::Matrix4<GMFloat>) {
        if self.points.is_empty() {
            return;
        }
        let global_mat = parent_matrix * self.model_matrix;
        let ts_transform = tiny_skia::Transform::from_row(
            global_mat.m11 as f32, global_mat.m21 as f32,
            global_mat.m12 as f32, global_mat.m22 as f32,
            global_mat.m14 as f32, global_mat.m24 as f32,
        );

        let mut pb = tiny_skia::PathBuilder::new();
        let mut first = true;
        for p in &self.points {
            let px = ctx.scene_config.convert_coord_x(p.x);
            let py = ctx.scene_config.convert_coord_y(p.y);
            if first {
                pb.move_to(px, py);
                first = false;
            } else {
                pb.line_to(px, py);
            }
        }
        if let Some(path) = pb.finish() {
            let mut paint = tiny_skia::Paint::default();
            paint.set_color(self.draw_config.color.into());

            if let Some(mut stroke) = self.draw_config.get_stroke(ctx.scene_config.scale_factor) {
                ctx.pixmap.stroke_path(
                    &path,
                    &paint,
                    &stroke,
                    ts_transform,
                    None,
                );
            }
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
