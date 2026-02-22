use std::{cell::RefCell, collections::VecDeque, rc::Rc, time::Instant};

use nalgebra::{Point3, Vector3};
use tiny_skia::Pixmap;

use crate::{
    mobjects::{text::Text, Mobject, MobjectClone, SimpleLine},
    video_backend::{ffmpeg::FfmpegBackend, FfmpegPipeBackend, FfmpegPipeEncoder},
    Context, GMFloat, Scene, SceneConfig,
};

trait Animation: Iterator<Item = Vec<u8>> {
    fn total_frame(&self) -> u32;
}

pub struct AnimationConfig {
    pub total_frame: u32,
    pub current_frame: u32,
    pub rate_function: fn(GMFloat) -> GMFloat,
}
pub struct SimpleMovement {
    pub displacement: Vector3<GMFloat>,
    pub scene: Rc<RefCell<Scene>>,
    pub ctx: Rc<RefCell<Context>>,
    pub m: Rc<RefCell<Box<dyn Mobject>>>,
    pub animation_config: AnimationConfig,
    pub last_progress: GMFloat,
}

pub struct MovementPrecise {
    pub displacement: Vector3<GMFloat>,
    pub scene: Rc<RefCell<Scene>>,
    pub ctx: Rc<RefCell<Context>>,
    pub m: Rc<RefCell<Box<dyn Mobject>>>,
    pub start_mobject: Rc<RefCell<Box<dyn MobjectClone>>>,
    pub animation_config: AnimationConfig,
}

impl Iterator for SimpleMovement {
    type Item = Vec<u8>;
    fn next(&mut self) -> Option<Self::Item> {
        self.animation_config.current_frame += 1;

        let current_frame = self.animation_config.current_frame;
        let total_frame = self.animation_config.total_frame;

        if current_frame > total_frame {
            return None;
        }

        let progress = (self.animation_config.rate_function)(
            current_frame as GMFloat / total_frame as GMFloat,
        );

        let delta_progress = progress - self.last_progress;
        self.last_progress = progress;

        let translation = nalgebra::Matrix4::new_translation(&(self.displacement * delta_progress));
        self.m
            .borrow_mut()
            .transform(nalgebra::Transform::from_matrix_unchecked(translation));
        // Some()
        for d in &self.scene.borrow().mobjects {
            self.ctx.borrow_mut().clear_transparent();
            d.borrow().draw(&mut *self.ctx.borrow_mut());
        }
        Some(self.ctx.borrow().image_bytes().to_vec())
    }
}

impl Animation for SimpleMovement {
    fn total_frame(&self) -> u32 {
        return self.animation_config.total_frame;
    }
}

impl SimpleMovement {
    pub fn new() {}
}

pub struct SimpleRotate {
    pub axisangle: Vector3<GMFloat>,
    pub point: Point3<GMFloat>,
    pub scene: Rc<RefCell<Scene>>,
    pub ctx: Rc<RefCell<Context>>,
    pub m: Rc<RefCell<Box<dyn Mobject>>>,
    pub animation_config: AnimationConfig,
    pub last_progress: GMFloat,
}

impl Iterator for SimpleRotate {
    type Item = Vec<u8>;
    fn next(&mut self) -> Option<Self::Item> {
        self.animation_config.current_frame += 1;

        let current_frame = self.animation_config.current_frame;
        let total_frame = self.animation_config.total_frame;
        if current_frame > total_frame {
            return None;
        }

        let progress = (self.animation_config.rate_function)(
            current_frame as GMFloat / total_frame as GMFloat,
        );

        let delta_progress = progress - self.last_progress;
        self.last_progress = progress;
        let rotation_matrix =
            nalgebra::Matrix4::new_rotation_wrt_point(self.axisangle * delta_progress, self.point);
        self.m
            .borrow_mut()
            .transform(nalgebra::Transform::from_matrix_unchecked(rotation_matrix));
        for d in &self.scene.borrow().mobjects {
            self.ctx.borrow_mut().clear_transparent();
            d.borrow().draw(&mut *self.ctx.borrow_mut());
        }
        Some(self.ctx.borrow().image_bytes().to_vec())
    }
}

#[test]
fn test_simple_move() {
    let mut ctx = Context::default();
    let mut scene = Scene::default();
    let mut line: Box<dyn Mobject> = Box::new(SimpleLine {
        p0: Point3::new(0.0, 0.0, 0.0),
        p1: Point3::new(1.0, 1.0, 0.0),
        draw_config: Default::default(),
    });
    let line_ref = Rc::new(RefCell::new(line));
    scene.add_ref(line_ref.clone());
    let scene = Rc::new(RefCell::new(scene));
    let ctx = Rc::new(RefCell::new(ctx));
    let simple_move = SimpleMovement {
        displacement: Vector3::new(2.0, 0.0, 0.0),
        scene: scene.clone(),
        ctx: ctx.clone(),
        m: line_ref.clone(),
        animation_config: AnimationConfig {
            total_frame: 60 * 30,
            current_frame: 0,
            rate_function: |x| x,
        },
        last_progress: 0.0,
    };
    use crate::video_backend::{
        ColorOrder, FfmpegPipeBackend, FrameMessage, VideoBackend, VideoBackendController,
        VideoBackendType, VideoConfig,
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
            FfmpegPipeEncoder::Libx264,
            false,
        )),
    };
    // let mut video_backend_controller = VideoBackendController::new(video_backend_var);
    for frame in simple_move {
        // video_backend_controller.write_frame(frame);
        video_backend_var.write_frame(&frame);
    }
    // video_backend_controller.end();
}

impl Animation for SimpleRotate {
    fn total_frame(&self) -> u32 {
        self.animation_config.total_frame
    }
}

pub struct Wait {
    pub scene: Rc<RefCell<Scene>>,
    pub ctx: Rc<RefCell<Context>>,
    pub animation_config: AnimationConfig,
    pub is_first_frame: bool,
}

impl Iterator for Wait {
    type Item = Vec<u8>;
    fn next(&mut self) -> Option<Self::Item> {
        self.animation_config.current_frame += 1;
        let current_frame = self.animation_config.current_frame;
        let total_frame = self.animation_config.total_frame;
        if current_frame > total_frame {
            return None;
        }
        if self.is_first_frame {
            for d in &self.scene.borrow().mobjects {
                self.ctx.borrow_mut().clear_transparent();
                d.borrow().draw(&mut *self.ctx.borrow_mut());
            }
            self.is_first_frame = false;
        }
        Some(self.ctx.borrow().image_bytes().to_vec())
    }
}

impl Animation for Wait {
    fn total_frame(&self) -> u32 {
        self.animation_config.total_frame
    }
}

#[test]
fn test_simple_rotate() {
    let width: usize = 1920;
    let height: usize = 1080;
    let mut ctx = Context {
        ctx_type: crate::ContextType::TinySKIA(Pixmap::new(width as u32, height as u32).unwrap()),
        scene_config: SceneConfig {
            width: 16.0,
            height: 9.0,
            output_width: width as u32,
            output_height: height as u32,
            scale_factor: height as GMFloat / 16.0,
        },
    };
    let mut scene = Scene::default();
    let mut line: Box<dyn Mobject> = Box::new(SimpleLine {
        p0: Point3::new(0.0, 0.0, 0.0),
        p1: Point3::new(1.0, 1.0, 0.0),
        draw_config: Default::default(),
    });
    let line_ref = Rc::new(RefCell::new(line));
    scene.add_ref(line_ref.clone());
    let scene = Rc::new(RefCell::new(scene));
    let ctx = Rc::new(RefCell::new(ctx));
    let simple_move = SimpleRotate {
        axisangle: Vector3::new(0.0, 0.0, 3.14),
        point: Point3::origin(),
        scene: scene.clone(),
        ctx: ctx.clone(),
        m: line_ref.clone(),
        animation_config: AnimationConfig {
            total_frame: 240,
            current_frame: 0,
            rate_function: |x| x,
        },
        last_progress: 0.0,
    };
    let wait = Wait {
        scene: scene.clone(),
        ctx: ctx.clone(),
        animation_config: AnimationConfig {
            total_frame: 60,
            current_frame: 0,
            rate_function: |x| x,
        },
        is_first_frame: true,
    };
    use crate::video_backend::{
        ColorOrder, FfmpegPipeBackend, FrameMessage, VideoBackend, VideoBackendType, VideoConfig,
    };

    let video_config = VideoConfig {
        filename: "output.mp4".to_owned(),
        framerate: 60,
        output_width: width as u32,
        output_height: height as u32,
        color_order: ColorOrder::Rgba,
    };
    let mut video_backend_var = VideoBackend {
        backend_type: VideoBackendType::Ffmpeg(FfmpegBackend::new(&video_config)),
        // backend_type: VideoBackendType::FfmpegPipe(FfmpegPipeBackend::new(
        //     &video_config,
        //     FfmpegPipeEncoder::Libx264,
        //     false,
        // )),
    };

    let mut frames: VecDeque<Vec<u8>> = VecDeque::new();
    for frame in simple_move {
        frames.push_back(frame);
    }
    for f in frames {
        video_backend_var.write_frame(&f);
    }
    video_backend_var.close();
}
