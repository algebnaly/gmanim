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
    fn is_pure(&self) -> bool {
        true
    }
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
            let mat =
                nalgebra::Matrix4::new_rotation_wrt_point(self.axis_angle * alpha, self.center);
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
    vulkan_renderer: Option<crate::vulkan::renderer::VulkanRenderer>,
    current_anim: Option<Box<dyn Animation>>,
    current_anim_frame: u32,
    pub total_cached_frames: u32,
    pub current_frame_global: u32,
    pub cached_nv12_data: Option<Vec<u8>>,
}

impl Timeline {
    pub fn new(scene: Scene, ctx: Context) -> Self {
        Self {
            scene,
            ctx,
            actions: VecDeque::new(),
            vulkan_renderer: None,
            current_anim: None,
            current_anim_frame: 0,
            total_cached_frames: 0,
            current_frame_global: 0,
            cached_nv12_data: None,
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
        if self.vulkan_renderer.is_none() {
            if let Some(vk_ctx) = pollster::block_on(crate::vulkan::context::VulkanContext::new()) {
                self.vulkan_renderer = Some(crate::vulkan::renderer::VulkanRenderer::new(
                    std::sync::Arc::new(vk_ctx),
                ));
            }
        }

        if let Some(renderer) = &mut self.vulkan_renderer {
            renderer.render_scene(&self.scene, &self.ctx.scene_config, None);
        }
    }

    pub fn nv12_image_bytes(&self) -> Option<&[u8]> {
        if let Some(renderer) = &self.vulkan_renderer {
            renderer.get_nv12_bytes()
        } else {
            None
        }
    }

    pub fn image_bytes(&self) -> Option<&[u8]> {
        if let Some(renderer) = &self.vulkan_renderer {
            renderer.get_rgba_bytes()
        } else {
            None
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════
