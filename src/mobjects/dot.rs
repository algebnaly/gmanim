use nalgebra::Point3;
use tiny_skia::{FillRule, LineCap, LineJoin, Paint, PathBuilder, Stroke};

use crate::{
    mobjects::{Draw, DrawConfig, Mobject, Transform},
    Color, Context, GMFloat,
};

pub struct Dot {
    position: Point3<GMFloat>,
    radius: GMFloat,
    color: Color,
    draw_config: DrawConfig,
}

impl Default for Dot {
    fn default() -> Self {
        Self {
            position: Point3::origin(),
            radius: 0.05,
            color: Color::default(),
            draw_config: DrawConfig::default(),
        }
    }
}

impl Dot {
    pub fn new(
        position: Point3<GMFloat>,
        radius: GMFloat,
        color: Color,
        draw_config: DrawConfig,
    ) -> Self {
        Self {
            position,
            radius,
            color,
            draw_config,
        }
    }
}

impl Draw for Dot {
    fn draw(&self, ctx: &mut Context) {
        match &mut ctx.ctx_type {
            crate::ContextType::TinySKIA(p) => {
                println!("Drawing dot");
                let scale_factor = ctx.scene_config.scale_factor;
                let mut pb = tiny_skia::PathBuilder::new();
                let path = PathBuilder::from_circle(
                    ctx.scene_config.convert_coord_x(self.position.x),
                    ctx.scene_config.convert_coord_y(self.position.y),
                    self.radius * scale_factor,
                )
                .unwrap();

                let mut stroke = Stroke::default();
                stroke.width = self.draw_config.stoke_width * scale_factor;
                stroke.line_cap = LineCap::Round;
                stroke.line_join = LineJoin::Round;
                let mut paint = Paint::default();
                paint.set_color(self.draw_config.color.into());
                paint.anti_alias = true;

                p.fill_path(
                    &path,
                    &paint,
                    FillRule::Winding,
                    tiny_skia::Transform::identity(),
                    None,
                );
            }
            _ => {}
        }
    }
}
impl Transform for Dot {
    fn transform(&mut self, transform: nalgebra::Transform3<GMFloat>) {
        self.position = transform.transform_point(&self.position);
    }
}

impl Mobject for Dot {}
