use crate::GMFloat;
use nalgebra::Point3;

use super::{DrawConfig, Mobject};

pub struct Text {
    pub text: String,
    pub position: Point3<GMFloat>,
    pub font_size: GMFloat,
    pub draw_config: DrawConfig,
}

impl Mobject for Text {
    fn default_name(&self) -> &'static str {
        "Text"
    }
}

impl Text {
    pub fn new(
        text: String,
        position: Point3<GMFloat>,
        font_size: GMFloat,
        draw_config: DrawConfig,
    ) -> Self {
        Text {
            text,
            position,
            font_size,
            draw_config,
        }
    }
}
