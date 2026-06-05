use nalgebra::Point3;
use tiny_skia::{FillRule, Paint, Shader};

use crate::{Color, Context, GMFloat, GMPoint, Scene};

use super::{Draw, DrawConfig, Mobject, Transform};

struct Polygon {
    vertices: Vec<GMPoint>,
    draw_config: DrawConfig,
}

impl Polygon {
    pub fn new(vertices: Vec<GMPoint>) -> Self {
        Self {
            vertices,
            draw_config: DrawConfig::default(),
        }
    }
}
impl Draw for Polygon {
    fn draw(&self, ctx: &mut crate::Context) {
        match &mut ctx.ctx_type {
            crate::ContextType::TinySKIA(pixmap) => {
                let mut pb = tiny_skia::PathBuilder::new();
                let mut v_list = self.vertices.iter();
                let start = v_list.next().unwrap();

                pb.move_to(
                    ctx.scene_config.convert_coord_x(start.x),
                    ctx.scene_config.convert_coord_y(start.y),
                );
                for p in v_list {
                    pb.line_to(
                        ctx.scene_config.convert_coord_x(p.x),
                        ctx.scene_config.convert_coord_y(p.y),
                    );
                }
                pb.close();
                let path = pb.finish().unwrap();
                let mut paint = Paint::default();
                paint.set_color(self.draw_config.color.into());
                pixmap.fill_path(
                    &path,
                    &paint,
                    FillRule::EvenOdd,
                    tiny_skia::Transform::identity(),
                    None,
                );
            }
            _ => {}
        }
    }
}

impl Transform for Polygon {
    fn transform(&mut self, transform: nalgebra::Transform3<GMFloat>) {
        for p in &mut self.vertices {
            *p = transform.transform_point(p);
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
