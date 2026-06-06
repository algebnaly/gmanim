use std::{cell::RefCell, collections::VecDeque, rc::Rc};

use nalgebra::{Point3, Vector3};

use crate::{
    mobjects::{Mobject, Transform},
    Context, GMFloat, Scene,
};

// ═══════════════════════════════════════════════════════════════════════════
// Animation trait
// ═══════════════════════════════════════════════════════════════════════════

/// An animation transforms mobject state over a normalized progress range [0, 1].
///
/// Implementors only need to define how objects change — rendering is handled
/// by [`Timeline`].
pub trait Animation {
    /// Total number of frames this animation spans.
    fn total_frames(&self) -> u32;

    /// Advance the animation to progress `t` ∈ [0, 1].
    ///
    /// Called once per frame with monotonically increasing `t` (after rate_function).
    /// For stateful animations that depend on the previous frame, track `last_t`
    /// internally and compute `dt = t - last_t`.
    fn update(&mut self, t: GMFloat, scene: &mut Scene);

    /// Called once before the first frame (optional).
    fn begin(&mut self, _scene: &mut Scene) {}

    /// Called once after the last frame (optional).
    fn finish(&mut self, _scene: &mut Scene) {}
}

// ═══════════════════════════════════════════════════════════════════════════
// AnimationConfig
// ═══════════════════════════════════════════════════════════════════════════

pub struct AnimationConfig {
    pub total_frames: u32,
    pub rate_function: fn(GMFloat) -> GMFloat,
}

impl AnimationConfig {
    /// Compute eased progress for a given frame number.
    pub fn progress(&self, frame: u32) -> GMFloat {
        let raw = frame as GMFloat / self.total_frames as GMFloat;
        (self.rate_function)(raw)
    }
}

impl Default for AnimationConfig {
    fn default() -> Self {
        Self {
            total_frames: 60,
            rate_function: |x| x,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Concrete animations
// ═══════════════════════════════════════════════════════════════════════════

/// Translate a mobject by a displacement vector over the animation duration.
pub struct Move {
    pub target: Rc<RefCell<Box<dyn Mobject>>>,
    pub displacement: Vector3<GMFloat>,
    pub config: AnimationConfig,
    last_t: GMFloat,
}

impl Move {
    pub fn new(
        target: Rc<RefCell<Box<dyn Mobject>>>,
        displacement: Vector3<GMFloat>,
        total_frames: u32,
    ) -> Self {
        Self {
            target,
            displacement,
            config: AnimationConfig {
                total_frames,
                ..Default::default()
            },
            last_t: 0.0,
        }
    }
}

impl Animation for Move {
    fn total_frames(&self) -> u32 {
        self.config.total_frames
    }

    fn update(&mut self, t: GMFloat, _scene: &mut Scene) {
        let dt = t - self.last_t;
        self.last_t = t;
        self.target.borrow_mut().move_this(self.displacement * dt);
    }
}

/// Rotate a mobject around a point over the animation duration.
pub struct Rotate {
    pub target: Rc<RefCell<Box<dyn Mobject>>>,
    pub axis_angle: Vector3<GMFloat>,
    pub center: Point3<GMFloat>,
    pub config: AnimationConfig,
    last_t: GMFloat,
}

impl Rotate {
    pub fn new(
        target: Rc<RefCell<Box<dyn Mobject>>>,
        axis_angle: Vector3<GMFloat>,
        center: Point3<GMFloat>,
        total_frames: u32,
    ) -> Self {
        Self {
            target,
            axis_angle,
            center,
            config: AnimationConfig {
                total_frames,
                ..Default::default()
            },
            last_t: 0.0,
        }
    }
}

impl Animation for Rotate {
    fn total_frames(&self) -> u32 {
        self.config.total_frames
    }

    fn update(&mut self, t: GMFloat, _scene: &mut Scene) {
        let dt = t - self.last_t;
        self.last_t = t;
        let mat = nalgebra::Matrix4::new_rotation_wrt_point(self.axis_angle * dt, self.center);
        self.target
            .borrow_mut()
            .apply_transform(mat);
    }
}

/// Hold the scene unchanged for a given number of frames.
pub struct Wait {
    pub config: AnimationConfig,
}

impl Wait {
    pub fn new(total_frames: u32) -> Self {
        Self {
            config: AnimationConfig {
                total_frames,
                ..Default::default()
            },
        }
    }
}

impl Animation for Wait {
    fn total_frames(&self) -> u32 {
        self.config.total_frames
    }
    fn update(&mut self, _t: GMFloat, _scene: &mut Scene) {}
}
/// An action in the timeline: either an animation or a one-shot scene script.
enum TimelineAction {
    Anim(Box<dyn Animation>),
    Script(Box<dyn FnOnce(&mut Scene)>),
}

/// Plays a sequence of animations, driving the render loop.
///
/// Timeline owns the [`Scene`] and [`Context`]. It calls each animation's
/// `update()` per frame, renders the scene, then passes the result to a
/// caller-provided callback.
///
/// Between animations, scene scripts can be inserted via [`Timeline::run`]
/// to add/remove mobjects or modify scene state.
pub struct Timeline {
    pub scene: Scene,
    pub ctx: Context,
    actions: VecDeque<TimelineAction>,
    wgpu_renderer: Option<crate::wgpu::renderer::WgpuRenderer>,
}

impl Timeline {
    pub fn new(scene: Scene, ctx: Context) -> Self {
        Self {
            scene,
            ctx,
            actions: VecDeque::new(),
            wgpu_renderer: None,
        }
    }

    /// Queue an animation to play.
    pub fn play(&mut self, anim: impl Animation + 'static) {
        self.actions.push_back(TimelineAction::Anim(Box::new(anim)));
    }

    /// Queue a scene modification script to run between animations.
    ///
    /// ```ignore
    /// timeline.play(Rotate::new(line_ref, ...));
    /// timeline.run(|scene| {
    ///     scene.add(Box::new(new_object));
    ///     scene.mobjects.remove(0);
    /// });
    /// timeline.play(Wait::new(60));
    /// ```
    pub fn run(&mut self, script: impl FnOnce(&mut Scene) + 'static) {
        self.actions
            .push_back(TimelineAction::Script(Box::new(script)));
    }

    /// Total number of frames across all queued animations.
    pub fn total_frames(&self) -> u32 {
        self.actions
            .iter()
            .map(|a| match a {
                TimelineAction::Anim(a) => a.total_frames(),
                TimelineAction::Script(_) => 0,
            })
            .sum()
    }

    /// Render all queued actions sequentially.
    ///
    /// `on_frame` is called once per rendered frame with a reference to the
    /// [`Context`] containing the finished pixel data. The caller decides
    /// what to do with it (encode, cache, display, etc.).
    pub fn render(&mut self, mut on_frame: impl FnMut(&Context)) {
        while let Some(action) = self.actions.pop_front() {
            match action {
                TimelineAction::Script(script) => {
                    script(&mut self.scene);
                }
                TimelineAction::Anim(mut anim) => {
                    let total = anim.total_frames();
                    anim.begin(&mut self.scene);

                    for frame in 1..=total {
                        let t = frame as GMFloat / total as GMFloat;

                        // 1. Animation updates mobject state
                        anim.update(t, &mut self.scene);

                        let output_w = self.ctx.scene_config.output_width as f32;
                        let output_h = self.ctx.scene_config.output_height as f32;

                        let (has_clip, clip_x, clip_y, clip_w, clip_h) = match self.scene.clip_rect {
                            Some(crate::ClipRect::Pixel(x, y, w, h)) => {
                                (true, x as f32, y as f32, w as f32, h as f32)
                            },
                            Some(crate::ClipRect::Logical(cx, cy, w, h)) => {
                                let (o_left, o_right, o_bottom, o_top) = self.scene.camera.ortho_params();
                                let log_w = o_right - o_left;
                                let log_h = o_top - o_bottom;
                                
                                let tl_x = cx - w / 2.0;
                                let tl_y = cy + h / 2.0;
                                
                                let norm_x = (tl_x - o_left) / log_w;
                                let norm_y = (o_top - tl_y) / log_h;
                                let norm_w = w / log_w;
                                let norm_h = h / log_h;
                                
                                (true, norm_x * output_w, norm_y * output_h, norm_w * output_w, norm_h * output_h)
                            },
                            None => (false, 0.0, 0.0, 0.0, 0.0),
                        };

                        // 2. Render 3D scene via WGPU
                        let mut primitives_3d = Vec::new();
                        for m in &self.scene.mobjects {
                            if let Some(obj_3d) = m.borrow().as_3d() {
                                primitives_3d.push(obj_3d.as_primitive_data());
                            }
                        }

                        if !primitives_3d.is_empty() {
                            if self.wgpu_renderer.is_none() {
                                if let Some(wgpu_ctx) =
                                    pollster::block_on(crate::wgpu::context::WgpuContext::new())
                                {
                                    self.wgpu_renderer =
                                        Some(crate::wgpu::renderer::WgpuRenderer::new(
                                            std::sync::Arc::new(wgpu_ctx),
                                        ));
                                }
                            }

                            if let Some(renderer) = &self.wgpu_renderer {
                                let width = self.ctx.scene_config.output_width as f32;
                                let height = self.ctx.scene_config.output_height as f32;
                                let look = self.scene.camera.look_at_dir();
                                let fov = self.scene.camera.fov();

                                let camera_uniform = crate::wgpu::renderer::CameraUniform {
                                    pos: [
                                        self.scene.camera.position.x as f32,
                                        self.scene.camera.position.y as f32,
                                        self.scene.camera.position.z as f32,
                                    ],
                                    _padding0: 0,
                                    look_at: [
                                        self.scene.camera.position.x as f32 + look.x as f32,
                                        self.scene.camera.position.y as f32 + look.y as f32,
                                        self.scene.camera.position.z as f32 + look.z as f32,
                                    ],
                                    _padding1: 0,
                                    up: [
                                        self.scene.camera.up_dir().x as f32,
                                        self.scene.camera.up_dir().y as f32,
                                        self.scene.camera.up_dir().z as f32,
                                    ],
                                    fov: fov as f32,
                                    width,
                                    height,
                                    proj_type: self.scene.camera.proj_type(),
                                    ortho_left: self.scene.camera.ortho_params().0 as f32,
                                    ortho_right: self.scene.camera.ortho_params().1 as f32,
                                    ortho_bottom: self.scene.camera.ortho_params().2 as f32,
                                    ortho_top: self.scene.camera.ortho_params().3 as f32,
                                    has_clip: if has_clip { 1 } else { 0 },
                                    clip_x,
                                    clip_y,
                                    clip_w,
                                    clip_h,
                                    _padding2: [0, 0, 0, 0],
                                };
                                renderer.render(
                                    self.ctx.scene_config.output_width,
                                    self.ctx.scene_config.output_height,
                                    &camera_uniform,
                                    &primitives_3d,
                                    self.ctx.pixmap.data_mut(),
                                );
                            } else {
                                self.ctx.clear_transparent();
                            }
                        } else {
                            self.ctx.clear_transparent();
                        }

                        for m in &self.scene.mobjects {
                            if m.borrow().as_3d().is_none() {
                                m.borrow().draw(&mut self.ctx, nalgebra::Matrix4::identity());
                            }
                        }

                        if has_clip {
                            let mut paint = tiny_skia::Paint::default();
                            paint.blend_mode = tiny_skia::BlendMode::Clear;
                            let width = self.ctx.scene_config.output_width as f32;
                            let height = self.ctx.scene_config.output_height as f32;
                            let cx = clip_x;
                            let cy = clip_y;
                            let cw = clip_w;
                            let ch = clip_h;
                            
                            // Top
                            if cy > 0.0 {
                                if let Some(r) = tiny_skia::Rect::from_xywh(0.0, 0.0, width, cy) {
                                    self.ctx.pixmap.fill_rect(r, &paint, tiny_skia::Transform::identity(), None);
                                }
                            }
                            // Bottom
                            if cy + ch < height {
                                if let Some(r) = tiny_skia::Rect::from_xywh(0.0, cy + ch, width, height - (cy + ch)) {
                                    self.ctx.pixmap.fill_rect(r, &paint, tiny_skia::Transform::identity(), None);
                                }
                            }
                            // Left
                            if cx > 0.0 {
                                if let Some(r) = tiny_skia::Rect::from_xywh(0.0, cy, cx, ch) {
                                    self.ctx.pixmap.fill_rect(r, &paint, tiny_skia::Transform::identity(), None);
                                }
                            }
                            // Right
                            if cx + cw < width {
                                if let Some(r) = tiny_skia::Rect::from_xywh(cx + cw, cy, width - (cx + cw), ch) {
                                    self.ctx.pixmap.fill_rect(r, &paint, tiny_skia::Transform::identity(), None);
                                }
                            }
                        }

                        // 4. Deliver frame to consumer
                        on_frame(&self.ctx);
                    }

                    anim.finish(&mut self.scene);
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        mobjects::SimpleLine,
        video_backend::{vaapi::FfmpegVaapiBackend, ColorOrder, VideoConfig},
        SceneConfig,
    };
    use tiny_skia::Pixmap;

    #[test]
    fn test_timeline_rotate_vaapi() {
        let width = 1920u32;
        let height = 1080u32;

        let ctx = Context {
            pixmap: Pixmap::new(width, height).unwrap(),
            scene_config: SceneConfig {
                width: 16.0,
                height: 9.0,
                output_width: width,
                output_height: height,
                scale_factor: height as GMFloat / 16.0,
            },
        };

        let mut scene = Scene::default();
        let line_ref: Rc<RefCell<Box<dyn Mobject>>> = Rc::new(RefCell::new(Box::new(SimpleLine {
            p0: Point3::new(0.0, 0.0, 0.0),
            p1: Point3::new(1.0, 1.0, 0.0),
            draw_config: Default::default(),
        })));
        scene.add_ref(line_ref.clone());

        // Add background lines
        for i in 1..5000 {
            let new_line: Box<dyn Mobject> = Box::new(SimpleLine {
                p0: Point3::new(i as GMFloat, 0.0, 0.0),
                p1: Point3::new(i as GMFloat + 1.0, 1.0, 0.0),
                draw_config: Default::default(),
            });
            scene.add(new_line);
        }

        let mut timeline = Timeline::new(scene, ctx);

        // Queue animations
        timeline.play(Rotate::new(
            line_ref.clone(),
            Vector3::new(0.0, 0.0, 3.14),
            Point3::origin(),
            1200,
        ));
        timeline.play(Wait::new(60));

        // Render to vaapi backend
        let video_config = VideoConfig {
            filename: "output.mp4".to_owned(),
            framerate: 60,
            output_width: width,
            output_height: height,
            color_order: ColorOrder::Rgba,
        };
        let mut backend = FfmpegVaapiBackend::new(&video_config);

        timeline.render(|ctx| {
            let mut buf = backend.acquire_buffer();
            ctx.copy_image_into(buf.as_mut_slice());
            backend.submit_frame(buf);
        });

        backend.finish().unwrap();
    }
}
