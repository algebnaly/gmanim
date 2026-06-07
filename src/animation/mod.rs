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
    /// Called once when the animation starts playing.
    fn begin(&mut self, _scene: &mut Scene) {}

    /// Called every frame to update the scene.
    /// `alpha` is the absolute progress from 0.0 to 1.0.
    fn update(&mut self, alpha: GMFloat, scene: &mut Scene);

    /// Called once when the animation completes.
    fn finish(&mut self, _scene: &mut Scene) {}

    /// Total number of frames this animation spans.
    fn total_frames(&self) -> u32;

    /// Whether this animation is a pure function of alpha.
    /// Pure animations can be fast-forwarded (O(1)).
    /// Incremental animations must be stepped frame-by-frame.
    fn is_pure(&self) -> bool { true }
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
    pub start_matrix: Option<nalgebra::Matrix4<GMFloat>>,
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
            start_matrix: None,
        }
    }
}

impl Animation for Move {
    fn begin(&mut self, _scene: &mut Scene) {
        self.start_matrix = Some(self.target.borrow().get_model_matrix());
    }

    fn total_frames(&self) -> u32 {
        self.config.total_frames
    }

    fn update(&mut self, alpha: GMFloat, _scene: &mut Scene) {
        if let Some(start) = self.start_matrix {
            let mat = nalgebra::Matrix4::new_translation(&(self.displacement * alpha));
            self.target.borrow_mut().set_model_matrix(mat * start);
        }
    }
}

/// Rotate a mobject around a point over the animation duration.
pub struct Rotate {
    pub target: Rc<RefCell<Box<dyn Mobject>>>,
    pub axis_angle: Vector3<GMFloat>,
    pub center: Point3<GMFloat>,
    pub config: AnimationConfig,
    pub start_matrix: Option<nalgebra::Matrix4<GMFloat>>,
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
            start_matrix: None,
        }
    }
}

impl Animation for Rotate {
    fn begin(&mut self, _scene: &mut Scene) {
        self.start_matrix = Some(self.target.borrow().get_model_matrix());
    }

    fn total_frames(&self) -> u32 {
        self.config.total_frames
    }

    fn update(&mut self, alpha: GMFloat, _scene: &mut Scene) {
        if let Some(start) = self.start_matrix {
            let mat = nalgebra::Matrix4::new_rotation_wrt_point(self.axis_angle * alpha, self.center);
            self.target.borrow_mut().set_model_matrix(mat * start);
        }
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
    current_anim: Option<Box<dyn Animation>>,
    current_anim_frame: u32,
    pub total_cached_frames: u32,
    pub current_frame_global: u32,
}

impl Timeline {
    pub fn new(scene: Scene, ctx: Context) -> Self {
        Self {
            scene,
            ctx,
            actions: VecDeque::new(),
            wgpu_renderer: None,
            current_anim: None,
            current_anim_frame: 0,
            total_cached_frames: 0,
            current_frame_global: 0,
        }
    }

    /// Queue an animation to play.
    pub fn play(&mut self, anim: impl Animation + 'static) {
        self.total_cached_frames += anim.total_frames();
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
        self.total_cached_frames
    }

    /// Render all queued actions sequentially.
    pub fn render(&mut self, mut on_frame: impl FnMut(&Context)) {
        while self.step_frame() {
            on_frame(&self.ctx);
        }
    }

    /// Advance the timeline by one frame. Returns false if no more frames.
    pub fn step_frame(&mut self) -> bool {
        let has_more = self.advance_frame();
        if has_more {
            self.render_current_state();
        }
        has_more
    }

    /// Advance the timeline state by one frame without rendering.
    pub fn advance_frame(&mut self) -> bool {
        let initial_frame = self.current_frame_global;
        self.seek_to_frame(initial_frame + 1);
        self.current_frame_global > initial_frame
    }

    /// Fast-forward or seek to a specific frame relative to the current state.
    pub fn seek_to_frame(&mut self, target_frame: u32) {
        while self.current_frame_global < target_frame {
            if self.current_anim.is_none() {
                if let Some(action) = self.actions.pop_front() {
                    match action {
                        TimelineAction::Script(script) => {
                            script(&mut self.scene);
                            continue;
                        }
                        TimelineAction::Anim(mut anim) => {
                            anim.begin(&mut self.scene);
                            self.current_anim = Some(anim);
                            self.current_anim_frame = 0;
                        }
                    }
                } else {
                    break;
                }
            }

            if let Some(anim) = &mut self.current_anim {
                let total = anim.total_frames();
                let remaining_in_anim = total - self.current_anim_frame;
                let frames_to_advance = target_frame - self.current_frame_global;

                if anim.is_pure() && frames_to_advance >= remaining_in_anim {
                    self.current_frame_global += remaining_in_anim;
                    self.current_anim_frame = total;
                    anim.update(1.0, &mut self.scene);
                    
                    if let Some(mut anim) = self.current_anim.take() {
                        anim.finish(&mut self.scene);
                    }
                } else if anim.is_pure() && frames_to_advance > 1 {
                    self.current_frame_global += frames_to_advance;
                    self.current_anim_frame += frames_to_advance;
                    let raw_t = self.current_anim_frame as GMFloat / total as GMFloat;
                    anim.update(raw_t, &mut self.scene);
                } else {
                    self.current_frame_global += 1;
                    self.current_anim_frame += 1;
                    let raw_t = self.current_anim_frame as GMFloat / total as GMFloat;
                    anim.update(raw_t, &mut self.scene);

                    if self.current_anim_frame >= total {
                        if let Some(mut anim) = self.current_anim.take() {
                            anim.finish(&mut self.scene);
                        }
                    }
                }
            }
        }
    }

    pub fn render_current_state(&mut self) {
        let output_w = self.ctx.scene_config.output_width as f32;
        let output_h = self.ctx.scene_config.output_height as f32;

        if self.wgpu_renderer.is_none() {
            if let Some(wgpu_ctx) = pollster::block_on(crate::wgpu::context::WgpuContext::new()) {
                self.wgpu_renderer = Some(crate::wgpu::renderer::WgpuRenderer::new(
                    std::sync::Arc::new(wgpu_ctx),
                ));
            }
        }

        if let Some(renderer) = &mut self.wgpu_renderer {
            renderer.render_scene(&self.scene, &self.ctx.scene_config, Some(self.ctx.pixmap.data_mut()));
        } else {
            self.ctx.clear_transparent();
        }

        for m in &self.scene.mobjects {
            if m.borrow().as_3d().is_none() {
                m.borrow().draw(&mut self.ctx, nalgebra::Matrix4::identity());
            }
        }

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
