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
    pub mobjects: Vec<Rc<RefCell<Box<dyn mobjects::Mobject>>>>,
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
    pub fn add(
        &mut self,
        mobject: Box<dyn mobjects::Mobject>,
    ) -> Rc<RefCell<Box<dyn mobjects::Mobject>>> {
        let rc = Rc::new(RefCell::new(mobject));
        self.mobjects.push(rc.clone());
        rc
    }
    pub fn add_ref(&mut self, mobject_ref: Rc<RefCell<Box<dyn mobjects::Mobject>>>) {
        self.mobjects.push(mobject_ref.clone());
    }
}

#[test]
fn test_simple_line_image() {
    use mobjects::SimpleLine;
    let mut ctx = Context::default();
    let mut scene = Scene::new();
    let simple_line = SimpleLine {
        p0: nalgebra::Point3::new(0.0, 0.0, 0.0),
        p1: nalgebra::Point3::new(1.0, 1.0, 0.0),
        ..Default::default()
    };
    let simple_line2 = SimpleLine {
        p0: nalgebra::Point3::new(1.0, 1.0, 0.0),
        p1: nalgebra::Point3::new(5.0, 2.0, 0.0),
        ..Default::default()
    };
    scene.add(Box::new(simple_line));
    scene.add(Box::new(simple_line2));
    let path = std::env::temp_dir().join("simple_line.png");
    scene.save_png(&mut ctx, path.to_str().unwrap());
}

#[test]
fn test_polyline_image() {
    use mobjects::PolyLine;
    let mut ctx = Context::default();
    let mut scene = Scene::new();
    let polyline = PolyLine {
        points: vec![
            nalgebra::Point3::new(0.0, 0.0, 0.0),
            nalgebra::Point3::new(3.5, 1.0, 0.0),
            nalgebra::Point3::new(3.5, 3.5, 0.0),
            nalgebra::Point3::new(4.0, 4.0, 0.0),
            nalgebra::Point3::new(6.0, 4.0, 0.0),
        ],
        ..Default::default()
    };
    scene.add(Box::new(polyline));
    let path = std::env::temp_dir().join("poly_line.png");
    scene.save_png(&mut ctx, path.to_str().unwrap());
}

#[test]
fn test_rectangle_image() {
    use mobjects::Rectangle;
    let mut ctx = Context::default();
    let mut scene = Scene::new();
    let rectangle = Rectangle {
        p0: nalgebra::Point3::new(0.0, 0.0, 0.0),
        p1: nalgebra::Point3::new(3.0, 0.0, 0.0),
        p2: nalgebra::Point3::new(3.0, 3.0, 0.0),
        p3: nalgebra::Point3::new(0.0, 3.0, 0.0),
        ..Default::default()
    };
    scene.add(Box::new(rectangle));
    let path = std::env::temp_dir().join("rectangle.png");
    scene.save_png(&mut ctx, path.to_str().unwrap());
}

#[test]
fn write_frame() {
    use mobjects::Rectangle;
    use std::sync::{Arc, Mutex};
    use std::thread;
    let mut ctx = Context::default();
    let mut scene = Scene::new();
    let rectangle = Rectangle {
        p0: nalgebra::Point3::new(0.0, 0.0, 0.0),
        p1: nalgebra::Point3::new(3.0, 0.0, 0.0),
        p2: nalgebra::Point3::new(3.0, 3.0, 0.0),
        p3: nalgebra::Point3::new(0.0, 3.0, 0.0),
        ..Default::default()
    };
    scene.add(Box::new(rectangle));

    use std::io::Write;
    use std::process::Command;
    use std::sync::mpsc;
    use std::sync::mpsc::{Receiver, Sender};

    use video_backend::{
        ColorOrder, FfmpegPipeBackend, VideoBackend, VideoBackendType, VideoConfig,
    };

    let path = std::env::temp_dir().join("output.mp4");
    let video_config = VideoConfig {
        filename: path.to_str().unwrap().to_owned(),
        framerate: 60,
        output_height: 1080,
        output_width: 1920,
        color_order: ColorOrder::Rgba,
    };
    let mut video_backend_var = VideoBackend {
        backend_type: VideoBackendType::FfmpegPipe(FfmpegPipeBackend::new(
            &video_config,
            video_backend::FfmpegPipeEncoder::HevcNvenc,
            false,
        )),
    };

    for _ in 0..480 {
        let now = std::time::Instant::now();
        let translation =
            nalgebra::Matrix4::new_translation(&nalgebra::Vector3::new(0.01, 0.0, 0.0));
        scene.mobjects[0].borrow_mut().apply_transform(translation);
        ctx.clear_transparent();
        for m in scene.mobjects.iter() {
            m.borrow().draw(&mut ctx, nalgebra::Matrix4::identity());
        }
        let mut buf = video_backend_var.acquire_buffer();
        ctx.copy_image_into(buf.as_mut_slice());
        video_backend_var.submit_frame(buf);
        println!("takes {:?}", now.elapsed());
    }
}
