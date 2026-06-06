use nalgebra::Point3;
use tiny_skia::{FillRule, LineCap, LineJoin, Paint, PathBuilder, Stroke};

use crate::{
    mobjects::{Draw, DrawConfig, Mobject, Transform},
    Color, Context, GMFloat,
};

pub struct Dot {
    pub position: Point3<GMFloat>,
    pub radius: GMFloat,
    pub color: Color,
    pub draw_config: DrawConfig,
    pub model_matrix: nalgebra::Matrix4<GMFloat>,
}

impl Default for Dot {
    fn default() -> Self {
        Self {
            position: Point3::origin(),
            radius: 0.05,
            color: Color::default(),
            draw_config: DrawConfig::default(),
            model_matrix: nalgebra::Matrix4::identity(),
        }
    }
}

impl Dot {
    pub fn new(position: Point3<GMFloat>, radius: GMFloat, color: Color, draw_config: DrawConfig) -> Self {
        Self {
            position,
            radius,
            color,
            draw_config,
            model_matrix: nalgebra::Matrix4::identity(),
        }
    }
}

impl Draw for Dot {
    fn draw(&self, ctx: &mut Context, parent_matrix: nalgebra::Matrix4<GMFloat>) {
        let global_mat = parent_matrix * self.model_matrix;
        let ts_transform = tiny_skia::Transform::from_row(
            global_mat.m11 as f32, global_mat.m21 as f32,
            global_mat.m12 as f32, global_mat.m22 as f32,
            global_mat.m14 as f32, global_mat.m24 as f32,
        );

        let path = tiny_skia::PathBuilder::from_circle(
            ctx.scene_config.convert_coord_x(self.position.x) as f32 * ctx.scene_config.scale_factor,
            ctx.scene_config.convert_coord_y(self.position.y) as f32 * ctx.scene_config.scale_factor,
            self.radius as f32 * ctx.scene_config.scale_factor, // scale radius
        )
        .unwrap();

        let mut paint = tiny_skia::Paint::default();
        paint.set_color(self.draw_config.color.into());
        ctx.pixmap.fill_path(
            &path,
            &paint,
            Default::default(),
            ts_transform,
            None,
        );
    }
}
impl Transform for Dot {
    fn get_model_matrix(&self) -> nalgebra::Matrix4<GMFloat> {
        self.model_matrix
    }
    fn set_model_matrix(&mut self, mat: nalgebra::Matrix4<GMFloat>) {
        self.model_matrix = mat;
    }
}

impl Mobject for Dot {}
