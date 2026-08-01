#![allow(unused)]

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
#[derive(Clone, Copy, Debug, PartialEq)]
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
    pub framerate: u32,
}

#[derive(Clone, Copy)]
pub struct RendererConfig {
    pub msaa_samples: u32,
    pub ssaa_factor: u32,
    pub output_color_profile: OutputColorProfile,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum OutputColorProfile {
    #[default]
    Bt709Sdr,
    Bt2020Pq,
    Bt2020Hlg,
}

impl OutputColorProfile {
    pub const fn is_hdr(self) -> bool {
        matches!(self, Self::Bt2020Pq | Self::Bt2020Hlg)
    }
}

impl Default for RendererConfig {
    fn default() -> Self {
        Self {
            msaa_samples: 8,
            ssaa_factor: 1,
            output_color_profile: OutputColorProfile::Bt709Sdr,
        }
    }
}

#[derive(Clone)]
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
        framerate: u32,
    ) -> Self {
        Context {
            scene_config: SceneConfig {
                width,
                height,
                output_width,
                output_height,
                scale_factor,
                framerate,
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
            framerate: 60,
        }
    }
}

impl Default for Context {
    fn default() -> Self {
        let scene_config = SceneConfig::default();
        Self { scene_config }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ClipRect {
    Logical(f32, f32, f32, f32), // center_x, center_y, width, height
    Pixel(u32, u32, u32, u32),   // top_left_x, top_left_y, width, height
}

#[derive(Clone)]
pub struct Scene {
    pub world: mobjects::SceneWorld,
    pub camera: camera::Camera,
    pub point_light: PointLight,
    pub environment_light: EnvironmentLight,
    pub clip_rect: Option<ClipRect>,
    pub aa_level: u32,
    pub background_color: Color,
}

#[derive(Clone)]
pub struct SceneSnapshot {
    world: mobjects::SceneWorld,
    camera: camera::Camera,
    point_light: PointLight,
    environment_light: EnvironmentLight,
    clip_rect: Option<ClipRect>,
    aa_level: u32,
    pub background_color: Color,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointLight {
    pub position: Point3<GMFloat>,
    pub color: Color,
    pub intensity: GMFloat,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EnvironmentLight {
    pub color: Color,
    pub intensity: GMFloat,
    pub rotation_radians: GMFloat,
}

impl Scene {
    pub fn snapshot(&self) -> SceneSnapshot {
        SceneSnapshot {
            world: self.world.clone(),
            camera: self.camera.clone(),
            point_light: self.point_light,
            environment_light: self.environment_light,
            clip_rect: self.clip_rect,
            aa_level: self.aa_level,
            background_color: self.background_color,
        }
    }

    pub fn restore(&mut self, snapshot: &SceneSnapshot) {
        self.world = snapshot.world.clone();
        self.camera = snapshot.camera.clone();
        self.point_light = snapshot.point_light;
        self.environment_light = snapshot.environment_light;
        self.clip_rect = snapshot.clip_rect;
        self.aa_level = snapshot.aa_level;
        self.background_color = snapshot.background_color;
    }

    pub fn from_snapshot(snapshot: &SceneSnapshot) -> Self {
        Self {
            world: snapshot.world.clone(),
            camera: snapshot.camera.clone(),
            point_light: snapshot.point_light,
            environment_light: snapshot.environment_light,
            clip_rect: snapshot.clip_rect,
            aa_level: snapshot.aa_level,
            background_color: snapshot.background_color,
        }
    }

    pub fn new() -> Self {
        Scene {
            world: mobjects::SceneWorld::default(),
            camera: camera::Camera::default(),
            point_light: PointLight {
                position: Point3::new(5.0, 5.0, 5.0),
                color: Color::new(255, 255, 255, 255),
                intensity: 180.0,
            },
            environment_light: EnvironmentLight {
                color: Color::new(115, 130, 145, 255),
                intensity: 0.12,
                rotation_radians: 0.0,
            },
            clip_rect: None,
            aa_level: 1,
            background_color: Color::new(0, 0, 0, 0),
        }
    }
}

impl SceneSnapshot {
    pub(crate) fn synchronize_identities_from(&mut self, scene: &Scene) {
        self.world.synchronize_identities_from(&scene.world);
    }
}

impl Default for Scene {
    fn default() -> Self {
        Scene::new()
    }
}

impl Scene {
    pub fn add(&mut self, mobject: impl mobjects::Mobject) -> mobjects::MobjectId {
        self.world.spawn(mobject)
    }

    pub fn add_named(
        &mut self,
        name: impl Into<String>,
        mobject: impl mobjects::Mobject,
    ) -> mobjects::MobjectId {
        self.world.spawn_named(name, mobject)
    }

    pub fn add_rectangle(&mut self, rectangle: mobjects::Rectangle) -> mobjects::MobjectId {
        self.world.spawn_rectangle(rectangle)
    }

    pub fn add_rectangle_named(
        &mut self,
        name: impl Into<String>,
        rectangle: mobjects::Rectangle,
    ) -> mobjects::MobjectId {
        self.world.spawn_rectangle_named(name, rectangle)
    }

    pub fn add_tree(&mut self, bundle: mobjects::NodeBundle) -> mobjects::MobjectId {
        self.world.spawn_tree(bundle)
    }
}
