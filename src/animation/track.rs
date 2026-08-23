use std::sync::Arc;

use crate::{GMFloat, Scene};

use super::{
    error::TimelineError,
    property::{Property, PropertyAddress, TrackValue},
    time::FrameRange,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Composition {
    Replace,
}

#[derive(Clone)]
pub enum Curve<T> {
    Linear { from: T, to: T },
    Sampled(Arc<[T]>),
}

impl<T> Curve<T> {
    pub fn linear(from: T, to: T) -> Self {
        Self::Linear { from, to }
    }

    pub fn sampled(values: impl Into<Arc<[T]>>) -> Self {
        Self::Sampled(values.into())
    }

    fn validate(&self, duration: u32) -> Result<(), TimelineError> {
        if let Self::Sampled(values) = self
            && values.len() != duration as usize + 1
        {
            return Err(TimelineError::SampleCount {
                expected: duration as usize + 1,
                actual: values.len(),
            });
        }
        Ok(())
    }
}

fn curve_value_at<T: TrackValue>(curve: &Curve<T>, frame: u32, duration: u32) -> T {
    match curve {
        Curve::Linear { from, to } => {
            let alpha = frame as GMFloat / duration as GMFloat;
            TrackValue::interpolate(from, to, alpha)
        }
        Curve::Sampled(values) => values[frame as usize].clone(),
    }
}

trait TrackEvaluator: Send + Sync {
    fn validate(&self, scene: &Scene, duration: u32) -> Result<(), TimelineError>;
    fn apply(&self, frame: u32, duration: u32, scene: &mut Scene) -> Result<(), TimelineError>;
}

struct TypedTrack<P: Property> {
    property: P,
    curve: Curve<P::Value>,
}

impl<P: Property> TrackEvaluator for TypedTrack<P> {
    fn validate(&self, scene: &Scene, duration: u32) -> Result<(), TimelineError> {
        self.property.read(scene)?;
        self.curve.validate(duration)
    }

    fn apply(&self, frame: u32, duration: u32, scene: &mut Scene) -> Result<(), TimelineError> {
        if !self.property.is_present(scene)? {
            return Ok(());
        }
        self.property
            .write(scene, curve_value_at(&self.curve, frame, duration))?;
        if frame == duration {
            self.property.finalize(scene)?;
        }
        Ok(())
    }
}

#[derive(Clone)]
pub(super) struct ClipTrack {
    address: PropertyAddress,
    composition: Composition,
    evaluator: Arc<dyn TrackEvaluator>,
}

impl ClipTrack {
    pub(super) fn new<P: Property>(property: P, curve: Curve<P::Value>) -> Self {
        Self {
            address: property.address(),
            composition: Composition::Replace,
            evaluator: Arc::new(TypedTrack { property, curve }),
        }
    }

    pub(super) fn validate(&self, scene: &Scene, duration: u32) -> Result<(), TimelineError> {
        self.evaluator.validate(scene, duration)
    }

    pub(super) fn compile(self, range: FrameRange) -> CompiledTrack {
        CompiledTrack {
            range,
            address: self.address,
            composition: self.composition,
            evaluator: self.evaluator,
        }
    }
}

#[derive(Clone)]
pub(super) struct CompiledTrack {
    range: FrameRange,
    address: PropertyAddress,
    composition: Composition,
    evaluator: Arc<dyn TrackEvaluator>,
}

impl CompiledTrack {
    pub(super) fn range(&self) -> FrameRange {
        self.range
    }

    pub(super) fn address(&self) -> PropertyAddress {
        self.address
    }

    pub(super) fn composition(&self) -> Composition {
        self.composition
    }

    pub(super) fn apply(&self, frame: u32, scene: &mut Scene) -> Result<(), TimelineError> {
        if frame <= self.range.start {
            return Ok(());
        }
        let local_frame = (frame - self.range.start).min(self.range.duration());
        self.evaluator
            .apply(local_frame, self.range.duration(), scene)
    }
}
