use nalgebra::Point3;
use tiny_skia::{FillRule, Paint, Shader};

use crate::{Color, Context, GMFloat, GMPoint, Scene};

use super::{Draw, DrawConfig, Mobject, Transform};

pub struct Polygon {
    pub vertices: Vec<GMPoint>,
    pub draw_config: DrawConfig,
    pub path: tiny_skia::Path,
    pub model_matrix: nalgebra::Matrix4<crate::GMFloat>,
}

impl Polygon {
    pub fn new(vertices: Vec<GMPoint>) -> Self {
        let mut pb = tiny_skia::PathBuilder::new();
        if !vertices.is_empty() {
            let mut v_list = vertices.iter();
            let start = v_list.next().unwrap();
            pb.move_to(start.x as f32, start.y as f32);
            for p in v_list {
                pb.line_to(p.x as f32, p.y as f32);
            }
            pb.close();
        }
        let path = pb.finish().unwrap_or_else(|| {
            let mut pb = tiny_skia::PathBuilder::new();
            pb.move_to(0.0, 0.0);
            pb.finish().unwrap()
        });

        Self {
            vertices,
            draw_config: DrawConfig::default(),
            path,
            model_matrix: nalgebra::Matrix4::identity(),
        }
    }
}

impl Transform for Polygon {
    fn get_model_matrix(&self) -> nalgebra::Matrix4<crate::GMFloat> {
        self.model_matrix
    }
    fn set_model_matrix(&mut self, mat: nalgebra::Matrix4<crate::GMFloat>) {
        self.model_matrix = mat;
    }
}

impl Draw for Polygon {
    fn draw(&self, ctx: &mut crate::Context, parent_matrix: nalgebra::Matrix4<crate::GMFloat>) {
        let global_mat = parent_matrix * self.model_matrix;
        let ts_transform = tiny_skia::Transform::from_row(
            global_mat.m11 as f32, global_mat.m21 as f32,
            global_mat.m12 as f32, global_mat.m22 as f32,
            global_mat.m14 as f32, global_mat.m24 as f32,
        );
        
        let mut paint = tiny_skia::Paint::default();
        paint.set_color(self.draw_config.color.into());

        if let Some(mut stroke) = self.draw_config.get_stroke(ctx.scene_config.scale_factor) {
            ctx.pixmap.stroke_path(
                &self.path,
                &paint,
                &stroke,
                ts_transform,
                None,
            );
        }

        if self.draw_config.fill {
            ctx.pixmap.fill_path(
                &self.path,
                &paint,
                tiny_skia::FillRule::EvenOdd,
                ts_transform,
                None,
            );
        }
    }
}

impl Mobject for Polygon {}

#[test]
pub fn test_polygon() {
    let mut ctx = Context::default();
    let mut scene = Scene::default();
    let v_list = vec![
        GMPoint::origin(),
        GMPoint::new(1.0, 1.0, 0.0),
        GMPoint::new(1.0, 2.0, 0.0),
    ];
    let mut polygon = Polygon::new(v_list);
    scene.add(Box::new(polygon));
    scene.save_png(&mut ctx, "output.png");
}
