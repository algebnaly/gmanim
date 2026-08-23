use std::sync::Arc;

use nalgebra::{Matrix4, Point3, Vector3};

use crate::{
    GMFloat, Scene,
    mobjects::{MobjectId, SpawnPlan},
};

use super::{
    error::TimelineError,
    operation::{ClipEvent, SceneOperation},
    property::{
        LayerProperty, Property, RectangleCornersProperty, TransformProperty, VisibilityProperty,
    },
    track::{ClipTrack, Curve},
};

#[derive(Clone)]
pub struct AnimationClip {
    duration: u32,
    tracks: Vec<ClipTrack>,
    events: Vec<ClipEvent>,
}

impl AnimationClip {
    pub fn new(duration: u32) -> Self {
        Self {
            duration,
            tracks: Vec::new(),
            events: Vec::new(),
        }
    }

    pub fn duration(&self) -> u32 {
        self.duration
    }

    pub fn track<P: Property>(mut self, property: P, curve: Curve<P::Value>) -> Self {
        self.tracks.push(ClipTrack::new(property, curve));
        self
    }

    pub fn transform(self, target: MobjectId, curve: Curve<Matrix4<GMFloat>>) -> Self {
        self.track(TransformProperty::new(target), curve)
    }

    pub fn rectangle_corners(self, target: MobjectId, curve: Curve<[Point3<GMFloat>; 4]>) -> Self {
        self.track(RectangleCornersProperty::new(target), curve)
    }

    pub fn visibility(self, target: MobjectId, curve: Curve<bool>) -> Self {
        self.track(VisibilityProperty::new(target), curve)
    }

    pub fn layer(self, target: MobjectId, curve: Curve<i32>) -> Self {
        self.track(LayerProperty::new(target), curve)
    }

    pub fn event(mut self, frame: u32, operation: SceneOperation) -> Self {
        self.events.push(ClipEvent::new(frame, operation));
        self
    }

    pub fn spawn(self, frame: u32, plan: SpawnPlan) -> Self {
        self.event(frame, SceneOperation::Spawn { plan })
    }

    pub fn write<P: Property>(self, frame: u32, property: P, value: P::Value) -> Self {
        self.event(frame, SceneOperation::write(property, value))
    }

    pub(super) fn into_parts(self) -> (u32, Vec<ClipTrack>, Vec<ClipEvent>) {
        (self.duration, self.tracks, self.events)
    }
}

pub trait AnimationBuilder {
    fn compile(self, scene: &Scene) -> Result<AnimationClip, TimelineError>;
}

impl AnimationBuilder for AnimationClip {
    fn compile(self, _scene: &Scene) -> Result<AnimationClip, TimelineError> {
        Ok(self)
    }
}

pub struct Move {
    target: MobjectId,
    displacement: Vector3<GMFloat>,
    duration: u32,
}

impl Move {
    pub fn new(target: MobjectId, displacement: Vector3<GMFloat>, duration: u32) -> Self {
        Self {
            target,
            displacement,
            duration,
        }
    }
}

impl AnimationBuilder for Move {
    fn compile(self, scene: &Scene) -> Result<AnimationClip, TimelineError> {
        if self.duration == 0 {
            return Err(TimelineError::EmptyClip);
        }
        let start = scene.world.get(self.target)?.transform();
        let values: Arc<[Matrix4<GMFloat>]> = (0..=self.duration)
            .map(|frame| {
                let alpha = frame as GMFloat / self.duration as GMFloat;
                Matrix4::new_translation(&(self.displacement * alpha)) * start
            })
            .collect::<Vec<_>>()
            .into();
        Ok(AnimationClip::new(self.duration).transform(self.target, Curve::Sampled(values)))
    }
}

pub struct Rotate {
    target: MobjectId,
    axis_angle: Vector3<GMFloat>,
    center: Point3<GMFloat>,
    duration: u32,
}

impl Rotate {
    pub fn new(
        target: MobjectId,
        axis_angle: Vector3<GMFloat>,
        center: Point3<GMFloat>,
        duration: u32,
    ) -> Self {
        Self {
            target,
            axis_angle,
            center,
            duration,
        }
    }
}

impl AnimationBuilder for Rotate {
    fn compile(self, scene: &Scene) -> Result<AnimationClip, TimelineError> {
        if self.duration == 0 {
            return Err(TimelineError::EmptyClip);
        }
        let start = scene.world.get(self.target)?.transform();
        let values: Arc<[Matrix4<GMFloat>]> = (0..=self.duration)
            .map(|frame| {
                let alpha = frame as GMFloat / self.duration as GMFloat;
                Matrix4::new_rotation_wrt_point(self.axis_angle * alpha, self.center) * start
            })
            .collect::<Vec<_>>()
            .into();
        Ok(AnimationClip::new(self.duration).transform(self.target, Curve::Sampled(values)))
    }
}

pub struct MorphRectangle {
    target: MobjectId,
    target_corners: [Point3<GMFloat>; 4],
    duration: u32,
}

impl MorphRectangle {
    pub fn new(target: MobjectId, target_corners: [Point3<GMFloat>; 4], duration: u32) -> Self {
        Self {
            target,
            target_corners,
            duration,
        }
    }
}

impl AnimationBuilder for MorphRectangle {
    fn compile(self, scene: &Scene) -> Result<AnimationClip, TimelineError> {
        if self.duration == 0 {
            return Err(TimelineError::EmptyClip);
        }
        let start = scene.world.rectangle(self.target)?.corners();
        Ok(AnimationClip::new(self.duration).rectangle_corners(
            self.target,
            Curve::Linear {
                from: start,
                to: self.target_corners,
            },
        ))
    }
}

pub struct Wait {
    duration: u32,
}

impl Wait {
    pub fn new(duration: u32) -> Self {
        Self { duration }
    }
}

impl AnimationBuilder for Wait {
    fn compile(self, _scene: &Scene) -> Result<AnimationClip, TimelineError> {
        Ok(AnimationClip::new(self.duration))
    }
}
