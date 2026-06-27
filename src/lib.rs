#![allow(unused)]

use std::cell::RefCell;
use std::rc::Rc;

use mobjects::{coordinate_change_x, coordinate_change_y};

pub mod camera;
use nalgebra::Point3;

pub mod animation;
pub mod log_utils;
pub mod math_utils;
pub mod mobjects;
pub mod video_backend;
pub mod vulkan;

cfg_if::cfg_if! {
    if #[cfg(feature = "gmfloat_f16")]{
        pub type GMFloat = f16;
    }else if #[cfg(feature = "gmfloat_f32")]{
        pub type GMFloat = f32;
    }else if #[cfg(feature = "gmfloat_f64")]{
        pub type GMFloat = f64;
    }else{
        pub type GMFloat = f32;
    }
}

pub type GMPoint = Point3<GMFloat>;
#[derive(Clone, Copy, Debug)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Color {
    pub fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }
    pub fn white() -> Self {
        Self::new(255, 255, 255, 255)
    }
    pub fn to_array(&self) -> [GMFloat; 4] {
        [
            self.r as GMFloat / 255.0,
            self.g as GMFloat / 255.0,
            self.b as GMFloat / 255.0,
            self.a as GMFloat / 255.0,
        ]
    }
}

impl Default for Color {
    fn default() -> Self {
        Self {
            r: 0x33,
            g: 0xcc,
            b: 0xff,
            a: 0xff,
        }
    }
}

#[derive(Clone, Debug, Copy)]
pub struct SceneConfig {
    pub width: GMFloat,
    pub height: GMFloat,
    pub output_width: u32,
    pub output_height: u32,
    pub scale_factor: GMFloat,
}

pub struct Context {
    pub scene_config: SceneConfig,
}

impl Context {
    pub fn new(
        width: GMFloat,
        height: GMFloat,
        output_width: u32,
        output_height: u32,
        scale_factor: GMFloat,
    ) -> Self {
        Context {
            scene_config: SceneConfig {
                width,
                height,
                output_width,
                output_height,
                scale_factor,
            },
        }
    }
}

impl SceneConfig {
    pub fn convert_coord_x(&self, x: GMFloat) -> GMFloat {
        coordinate_change_x(x, self.width) * self.scale_factor
    }
    pub fn convert_coord_y(&self, y: GMFloat) -> GMFloat {
        coordinate_change_y(y, self.height) * self.scale_factor
    }
}

impl Default for SceneConfig {
    fn default() -> Self {
        SceneConfig {
            width: 16.0,
            height: 9.0,
            output_width: 1920,
            output_height: 1080,
            scale_factor: 1920.0 / 16.0,
        }
    }
}

impl Default for Context {
    fn default() -> Self {
        let scene_config = SceneConfig::default();
        Self { scene_config }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum ClipRect {
    Logical(f32, f32, f32, f32), // center_x, center_y, width, height
    Pixel(u32, u32, u32, u32),   // top_left_x, top_left_y, width, height
}

pub struct Scene {
    pub mobjects: Vec<mobjects::MobjectRef>,
    pub camera: camera::Camera,
    pub light_pos: Point3<GMFloat>,
    pub light_color: Color,
    pub clip_rect: Option<ClipRect>,
    pub aa_level: u32,
}

impl Scene {
    pub fn new() -> Self {
        Scene {
            mobjects: Vec::new(),
            camera: camera::Camera::default(),
            light_pos: Point3::new(5.0, 5.0, 5.0),
            light_color: Color::new(255, 255, 255, 255),
            clip_rect: None,
            aa_level: 1,
        }
    }
}

impl Default for Scene {
    fn default() -> Self {
        Scene::new()
    }
}

impl Scene {
    pub fn add(&mut self, mobject: impl mobjects::Mobject + 'static) {
        self.mobjects
            .push(std::rc::Rc::new(std::cell::RefCell::new(mobject)));
    }

    pub fn add_ref(&mut self, mobject: mobjects::MobjectRef) {
        self.mobjects.push(mobject);
    }
}
