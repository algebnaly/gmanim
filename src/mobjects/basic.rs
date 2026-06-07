use nalgebra::Point3;
use std::f32::consts::PI;

use crate::{math_utils::k_for_bezier_arc, Color, Context, GMFloat, Scene};

use super::{get_2d_transform, Draw, DrawConfig, Mobject, Transform};

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
        let ts_transform = get_2d_transform(ctx, global_mat);

        let mut pb = tiny_skia::PathBuilder::new();
        pb.move_to(
            (self.p0.x) as f32,
            (self.p0.y) as f32,
        );
        pb.line_to(
            (self.p1.x) as f32,
            (self.p1.y) as f32,
        );
        pb.line_to(
            (self.p2.x) as f32,
            (self.p2.y) as f32,
        );
        pb.line_to(
            (self.p3.x) as f32,
            (self.p3.y) as f32,
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
        let ts_transform = get_2d_transform(ctx, global_mat);

        let mut pb = tiny_skia::PathBuilder::new();
        pb.move_to(
            (self.p0.x) as f32,
            (self.p0.y) as f32,
        );
        pb.line_to(
            (self.p1.x) as f32,
            (self.p1.y) as f32,
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
        let ts_transform = get_2d_transform(ctx, global_mat);

        let mut pb = tiny_skia::PathBuilder::new();
        
        let num_curves = ((self.end_angle - self.start_angle).abs() / (PI / 2.0)).ceil() as usize;
        if num_curves == 0 { return; }

        let angle_step = (self.end_angle - self.start_angle) / (num_curves as f32);
        
        let start_x = self.center_point.x + self.radius * self.start_angle.cos();
        let start_y = self.center_point.y + self.radius * self.start_angle.sin();
        pb.move_to(
            (start_x) as f32,
            (start_y) as f32
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
                (cp1_x) as f32, (cp1_y) as f32,
                (cp2_x) as f32, (cp2_y) as f32,
                (end_x) as f32, (end_y) as f32
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
        let ts_transform = get_2d_transform(ctx, global_mat);

        let mut pb = tiny_skia::PathBuilder::new();
        let mut first = true;
        for p in &self.points {
            let px = (p.x) as f32;
            let py = (p.y) as f32;
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

#[test]
fn test_draw_arc() {
    let mut ctx = Context::default();
    let mut scene = Scene::default();
    let arc = Arc::new(Point3::new(0.0, 1.0, 0.0), 0.0, PI as GMFloat * 2.0, 3.0);
    scene.add(Box::new(arc));
    scene.save_png(&mut ctx, "arc.png");
}
