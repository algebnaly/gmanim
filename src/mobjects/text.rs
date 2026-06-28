use std::fs;
use std::io::Read;

use log::info;
use rusttype::{Font, Scale, point};

use crate::log_utils::setup_logger;
use crate::mobjects::Draw;
use crate::{GMFloat, log_utils};
use nalgebra::{Point2, Point3, Vector3};

use super::path::PathElement;
use super::{DrawConfig, Mobject, Transform, coordinate_change_x, coordinate_change_y};

pub struct Text {
    pub base: crate::mobjects::MobjectBase,
    pub text: String,
    glyph_paths: Vec<GlyphPath>,
    pub position: Point3<GMFloat>,
    pub font_size: GMFloat,
    pub draw_config: DrawConfig,
}

impl PathElement {
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

struct GlyphPath {
    glyph_position: Point2<GMFloat>,
    path_elements: Vec<PathElement>,
}

impl GlyphPath {
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
    fn draw(&self, _ctx: &mut crate::Context, _parent_matrix: nalgebra::Matrix4<crate::GMFloat>) {
        // GPU tessellation logic will be added here
    }
}

impl Mobject for Text {
    fn base(&self) -> &crate::mobjects::MobjectBase {
        &self.base
    }
    fn base_mut(&mut self) -> &mut crate::mobjects::MobjectBase {
        &mut self.base
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
                base: crate::mobjects::MobjectBase::new("Text"),
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
            base: crate::mobjects::MobjectBase::new("Text"),
            text,
            glyph_paths,
            position,
            font_size,
            draw_config,
        }
    }
}
