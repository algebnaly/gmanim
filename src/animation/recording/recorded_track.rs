use std::sync::Arc;

use crate::Scene;

use super::super::{
    authoring::AnimationClip,
    property::{Property, PropertyAddress, PropertyError},
    track::Curve,
};

pub(super) trait ErasedRecordProperty: Send + Sync {
    fn address(&self) -> PropertyAddress;
    fn start_track(
        &self,
        baseline: &Scene,
        prefix_samples: u32,
        current: &Scene,
    ) -> Result<Box<dyn RecordedTrack>, PropertyError>;
}

pub(super) struct RecordProperty<P: Property>(P);

impl<P: Property> RecordProperty<P> {
    pub(super) fn new(property: P) -> Self {
        Self(property)
    }
}

impl<P: Property> ErasedRecordProperty for RecordProperty<P> {
    fn address(&self) -> PropertyAddress {
        self.0.address()
    }

    fn start_track(
        &self,
        baseline: &Scene,
        prefix_samples: u32,
        current: &Scene,
    ) -> Result<Box<dyn RecordedTrack>, PropertyError> {
        let baseline_value = self.0.read(baseline)?;
        let mut values = vec![baseline_value; prefix_samples as usize];
        values.push(self.0.read(current)?);
        Ok(Box::new(TypedRecordedTrack {
            property: self.0.clone(),
            values,
        }))
    }
}

pub(super) trait RecordedTrack: Send {
    fn push_from(&mut self, scene: &Scene) -> Result<(), PropertyError>;
    fn append_to(self: Box<Self>, clip: AnimationClip) -> AnimationClip;
}

struct TypedRecordedTrack<P: Property> {
    property: P,
    values: Vec<P::Value>,
}

impl<P: Property> RecordedTrack for TypedRecordedTrack<P> {
    fn push_from(&mut self, scene: &Scene) -> Result<(), PropertyError> {
        self.values.push(self.property.read(scene)?);
        Ok(())
    }

    fn append_to(self: Box<Self>, clip: AnimationClip) -> AnimationClip {
        clip.track(self.property, Curve::Sampled(Arc::from(self.values)))
    }
}
