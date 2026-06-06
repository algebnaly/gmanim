use std::fs;
use std::io::Read;

use log::info;
use rusttype::{point, Font, Scale};

use crate::log_utils::setup_logger;
use crate::mobjects::Draw;
use crate::{log_utils, GMFloat};
use nalgebra::{Point2, Point3, Vector3};

use super::path::PathElement;
use super::{coordinate_change_x, coordinate_change_y, DrawConfig, Mobject, Transform};

pub struct Text {
    pub text: String,
    glyph_paths: Vec<GlyphPath>,
    pub position: Point3<GMFloat>,
    pub font_size: GMFloat,
    pub draw_config: DrawConfig,
}

impl Transform for PathElement {
    fn transform(&mut self, transform: nalgebra::Transform3<GMFloat>) {
        match self {
            PathElement::MoveTo(p) => {
                *p = transform * p.clone();
            }
            PathElement::LineTo(p) => {
                *p = transform * p.clone();
            }
            PathElement::QuadTo(p1, p2) => {
                *p1 = transform * p1.clone();
                *p2 = transform * p2.clone();
            }
            PathElement::CubicTo(p1, p2, p3) => {
                *p1 = transform * p1.clone();
                *p2 = transform * p2.clone();
                *p3 = transform * p3.clone();
            }
            PathElement::Close => {}
        }
    }
}

pub enum FontConfig {
    Default,
    FontName(String),
    FontFile(String),
}

impl Transform for Text {
    fn transform(&mut self, transform: nalgebra::Transform3<GMFloat>) {
        for g in &mut self.glyph_paths {
            g.transform(transform);
        }
    }
}

struct GlyphPath {
    glyph_position: Point2<GMFloat>,
    path_elements: Vec<PathElement>,
}

impl Transform for GlyphPath {
    fn transform(&mut self, transform: nalgebra::Transform3<GMFloat>) {
        for p in &mut self.path_elements {
            p.transform(transform);
        }
    }
}

impl GlyphPath {
    fn new(position: rusttype::Point<f32>) -> Self {
        Self {
            glyph_position: Point2::new(
                position.x * SCALE_TEXT_FACTOR,
                -position.y * SCALE_TEXT_FACTOR,
            ),
            path_elements: vec![],
        }
    }
}

pub const SCALE_TEXT_FACTOR: f32 = 0.1;

impl rusttype::OutlineBuilder for GlyphPath {
    fn move_to(&mut self, x: f32, y: f32) {
        self.path_elements
            .push(PathElement::MoveTo(nalgebra::Point3::new(
                x * SCALE_TEXT_FACTOR + self.glyph_position.x,
                -y * SCALE_TEXT_FACTOR + self.glyph_position.y,
                0.0,
            )))
    }
    fn line_to(&mut self, x: f32, y: f32) {
        self.path_elements
            .push(PathElement::LineTo(nalgebra::Point3::new(
                x * SCALE_TEXT_FACTOR + self.glyph_position.x,
                -y * SCALE_TEXT_FACTOR + self.glyph_position.y,
                0.0,
            )))
    }
    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        self.path_elements.push(PathElement::QuadTo(
            nalgebra::Point3::new(
                x1 * SCALE_TEXT_FACTOR + self.glyph_position.x,
                -y1 * SCALE_TEXT_FACTOR + self.glyph_position.y,
                0.0,
            ),
            nalgebra::Point3::new(
                x * SCALE_TEXT_FACTOR + self.glyph_position.x,
                -y * SCALE_TEXT_FACTOR + self.glyph_position.y,
                0.0,
            ),
        ))
    }
    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        self.path_elements.push(PathElement::CubicTo(
            nalgebra::Point3::new(
                x1 * SCALE_TEXT_FACTOR + self.glyph_position.x,
                -y1 * SCALE_TEXT_FACTOR + self.glyph_position.y,
                0.0,
            ),
            nalgebra::Point3::new(
                x2 * SCALE_TEXT_FACTOR + self.glyph_position.x,
                -y2 * SCALE_TEXT_FACTOR + self.glyph_position.y,
                0.0,
            ),
            nalgebra::Point3::new(
                x * SCALE_TEXT_FACTOR + self.glyph_position.x,
                -y * SCALE_TEXT_FACTOR + self.glyph_position.y,
                0.0,
            ),
        ))
    }
    fn close(&mut self) {
        self.path_elements.push(PathElement::Close)
    }
}

impl Draw for Text {
    fn draw(&self, ctx: &mut crate::Context) {
        if self.text.len() == 0 {
            return; //this is no text to draw
        }
        let scale_factor = ctx.scene_config.scale_factor;
        for g in &self.glyph_paths {
            if g.path_elements.is_empty() {
                continue;
            }
            let mut pb = tiny_skia::PathBuilder::new();
            for path in &g.path_elements {
                match path {
                    PathElement::MoveTo(p) => {
                        let x = coordinate_change_x(p.x + self.position.x, ctx.scene_config.width)
                            as f32
                            * scale_factor as f32;
                        let y = coordinate_change_y(p.y + self.position.y, ctx.scene_config.height)
                            as f32
                            * scale_factor as f32;
                        pb.move_to(x, y);
                    }
                    PathElement::LineTo(p) => {
                        let x = coordinate_change_x(p.x + self.position.x, ctx.scene_config.width)
                            as f32
                            * scale_factor as f32;
                        let y = coordinate_change_y(p.y + self.position.y, ctx.scene_config.height)
                            as f32
                            * scale_factor as f32;
                        pb.line_to(x, y);
                    }
                    PathElement::QuadTo(p1, p2) => {
                        let x1 = coordinate_change_x(p1.x + self.position.x, ctx.scene_config.width)
                            as f32
                            * scale_factor as f32;
                        let y1 =
                            coordinate_change_y(p1.y + self.position.y, ctx.scene_config.height)
                                as f32
                                * scale_factor as f32;
                        let x2 = coordinate_change_x(p2.x + self.position.x, ctx.scene_config.width)
                            as f32
                            * scale_factor as f32;
                        let y2 =
                            coordinate_change_y(p2.y + self.position.y, ctx.scene_config.height)
                                as f32
                                * scale_factor as f32;
                        pb.quad_to(x1, y1, x2, y2);
                    }
                    PathElement::CubicTo(p1, p2, p3) => {
                        let x1 = coordinate_change_x(p1.x + self.position.x, ctx.scene_config.width)
                            as f32
                            * scale_factor as f32;
                        let y1 =
                            coordinate_change_y(p1.y + self.position.y, ctx.scene_config.height)
                                as f32
                                * scale_factor as f32;
                        let x2 = coordinate_change_x(p2.x + self.position.x, ctx.scene_config.width)
                            as f32
                            * scale_factor as f32;
                        let y2 =
                            coordinate_change_y(p2.y + self.position.y, ctx.scene_config.height)
                                as f32
                                * scale_factor as f32;
                        let x3 = coordinate_change_x(p3.x + self.position.x, ctx.scene_config.width)
                            as f32
                            * scale_factor as f32;
                        let y3 =
                            coordinate_change_y(p3.y + self.position.y, ctx.scene_config.height)
                                as f32
                                * scale_factor as f32;

                        pb.cubic_to(x1, y1, x2, y2, x3, y3);
                    }
                    PathElement::Close => {
                        pb.close();
                    }
                }
            }
            let path = pb.finish().unwrap();
            let mut stroke = tiny_skia::Stroke::default();
            stroke.width = self.draw_config.stoke_width * scale_factor;
            stroke.line_cap = tiny_skia::LineCap::Round;
            let mut paint = tiny_skia::Paint::default();
            paint.set_color(self.draw_config.color.into());
            ctx.pixmap.fill_path(
                &path,
                &paint,
                Default::default(),
                tiny_skia::Transform::identity(),
                None,
            );
        }
    }
}

impl Text {
    pub fn new(
        text: String,
        position: Point3<GMFloat>,
        font_size: GMFloat,
        draw_config: DrawConfig,
    ) -> Self {
        let mut glyph_paths = vec![];
        if text.len() == 0 {
            info!("text len is 0");
            return Text {
                text,
                glyph_paths,
                position,
                font_size,
                draw_config,
            };
        }
        let mut f = fs::File::open("/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc")
            .expect("can't open font file"); //replace with some font search
        let mut font_data_data = vec![];
        f.read_to_end(&mut font_data_data)
            .expect("can't read font file");

        let font =
            Font::try_from_bytes(&font_data_data).expect("failed to parse font file content");
        let scale = Scale::uniform(font_size as f32);
        let v_metrics = font.v_metrics(scale);
        // to see why we make start at (0.0, v_metrics.ascent), take a look at documentation
        let glyphs: Vec<_> = font
            .layout(&text, scale, point(0.0, 0.0 + v_metrics.ascent)) // maybe I need some padding here
            .collect();

        let img_height = (v_metrics.ascent - v_metrics.descent).ceil() as usize;
        let (img_width, min_x) = {
            let min_x = glyphs
                .first()
                .map(|g| g.pixel_bounding_box().unwrap().min.x)
                .unwrap();
            let max_x = glyphs
                .last()
                .map(|g| g.pixel_bounding_box().unwrap().max.x)
                .unwrap();
            ((max_x - min_x) as usize, min_x)
        }; // great, rusttype help me to calculate advance width and Kerning Pair
        for glyph in glyphs {
            let mut glyph_path = GlyphPath::new(glyph.position());
            glyph.build_outline(&mut glyph_path);
            glyph_paths.push(glyph_path);
        }
        Text {
            text,
            glyph_paths,
            position,
            font_size,
            draw_config,
        }
    }
}

impl Mobject for Text {}

#[test]
fn test_draw_text() {
    setup_logger().unwrap();
    let mut ctx = crate::Context::default();
    let mut scene = crate::Scene::new();
    let mut text = Text::new(
        "你好呀".to_owned(),
        Point3::new(0.0, 1.0, 0.0),
        32.0,
        DrawConfig::default(),
    );
    let rotation = nalgebra::Matrix4::new_rotation_wrt_point(
        nalgebra::Vector3::new(0.0, 0.0, 1.0),
        nalgebra::Point3::new(0.0, 0.0, 0.0),
    );
    text.transform(nalgebra::Transform::from_matrix_unchecked(rotation));
    text.draw(&mut ctx);
    ctx.pixmap.save_png("text_render.png");
}
