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
struct Color {
    r: u8,
    g: u8,
    b: u8,
    a: u8,
}

impl Color {
    fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }
}

impl From<Color> for tiny_skia::Color {
    fn from(value: Color) -> Self {
        Self::from_rgba8(value.r, value.g, value.b, value.a)
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

pub enum ContextType {
    TinySKIA(tiny_skia::Pixmap), // we always have cairo as a fallback
    VULKAN,
    CUDA,
    HIP,
}

pub struct SceneConfig {
    pub width: GMFloat,
    pub height: GMFloat,
    pub output_width: u32,
    pub output_height: u32,
    pub scale_factor: GMFloat,
}

pub struct Context {
    pub ctx_type: ContextType,
    pub scene_config: SceneConfig,
}

impl SceneConfig{
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
        let pixmap =
            tiny_skia::Pixmap::new(scene_config.output_width, scene_config.output_height).unwrap();
        Self {
            ctx_type: ContextType::TinySKIA(pixmap),
            scene_config,
        }
    }
}

impl Context {
    fn clear_transparent(&mut self) {
        match &mut self.ctx_type {
            ContextType::TinySKIA(pixmap) => {
                pixmap.fill(tiny_skia::Color::from_rgba8(0, 0, 0, 0xff));
            }
            _ => {}
        }
    }

    fn image_bytes(&self) -> &[u8] {
        match &self.ctx_type {
            ContextType::TinySKIA(pixmap) => pixmap.data(),
            _ => &[],
        }
    }


}

#[derive(Default)]
pub struct Scene {
    pub mobjects: Vec<Rc<RefCell<Box<dyn mobjects::Mobject>>>>,
}

impl Scene {
    pub fn new() -> Self {
        Scene { mobjects: vec![] }
    }
    pub fn save_png(&self, ctx: &mut Context, file_path: &str) {
        ctx.clear_transparent();

        for m in self.mobjects.iter() {
            m.borrow().draw(ctx);
        }

        match &mut ctx.ctx_type {
            ContextType::TinySKIA(pixmap) => {
                pixmap.save_png(file_path);
            }
            _ => {}
        }
    }

    pub fn add(&mut self, mobject: Box<dyn mobjects::Mobject>) {
        self.mobjects.push(Rc::new(RefCell::new(mobject)));
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
    scene.save_png(&mut ctx, "simple_line.png");
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
    scene.save_png(&mut ctx, "poly_line.png");
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
    scene.save_png(&mut ctx, "rectangle.png");
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
        ColorOrder, FfmpegPipeBackend, FrameMessage, VideoBackend, VideoBackendType, VideoConfig,
    };

    let video_config = VideoConfig {
        filename: "output.mp4".to_owned(),
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
        let translation = nalgebra::Transform3::<GMFloat>::from_matrix_unchecked(translation);
        scene.mobjects[0].borrow_mut().transform(translation);
        ctx.clear_transparent();
        for m in scene.mobjects.iter() {
            m.borrow().draw(&mut ctx);
        }
        video_backend_var.write_frame(ctx.image_bytes());
        println!("takes {:?}", now.elapsed());
    }
}

#[test]
fn thread_frame_pass() {
    use mobjects::Rectangle;
    use std::sync::mpsc::channel;
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

    use std::collections::VecDeque;
    use std::io::Write;
    use std::process::Command;
    use std::sync::mpsc;
    use std::sync::mpsc::{Receiver, Sender};

    use video_backend::{
        ColorOrder, FfmpegConfig,FfmpegPipeBackend, FrameMessage, VideoBackend, VideoBackendType, VideoConfig,
    };

    let video_config = VideoConfig {
        filename: "output.mp4".to_owned(),
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

    let (tx, rx) = channel::<video_backend::FrameMessage>();
    let queue = Arc::new(Mutex::new(VecDeque::<Vec<u8>>::new()));
    let queue_ref = queue.clone();

    let state = Arc::new(Mutex::new(video_backend::VideoBackendState::Running));
    let state_ref = state.clone();
    // let thread_handler = thread::spawn(move || {
    //     video_backend_var.write_frame_background(rx, state_ref, queue_ref);
    // });
    
    for _ in 0..480 {
        let now = std::time::Instant::now();
        let translation =
            nalgebra::Matrix4::new_translation(&nalgebra::Vector3::new(0.01, 0.0, 0.0));
        let translation = nalgebra::Transform3::<GMFloat>::from_matrix_unchecked(translation);
        scene.mobjects[0].borrow_mut().transform(translation);
        ctx.clear_transparent();
        for m in scene.mobjects.iter() {
            m.borrow().draw(&mut ctx);
        }
        let data_bytes = ctx.image_bytes().to_vec();
        {
            let mut queue_guard = queue.lock().unwrap();
            queue_guard.push_back(data_bytes);
        } //release queue lock
        {
            let mut state_guard = state.lock().unwrap();
            match *state_guard {
                video_backend::VideoBackendState::Sleeping => {
                    *state_guard = video_backend::VideoBackendState::Running;
                    tx.send(FrameMessage::Frame);
                }
                _ => {}
            }
        } //release state lock
        tx.send(FrameMessage::End);
    }
    // thread_handler.join();
}
