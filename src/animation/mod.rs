mod authoring;
mod compiler;
mod error;
mod operation;
mod property;
mod recording;
mod runtime;
mod time;
mod track;

pub use authoring::{AnimationBuilder, AnimationClip, MorphRectangle, Move, Rotate, Wait};
pub use compiler::TimelineBuilder;
pub use error::TimelineError;
pub use operation::SceneOperation;
pub use property::{
    AaLevelProperty, CameraPoseProperty, CameraProjectionProperty, EnvironmentLightProperty,
    ErasedProperty, LayerProperty, PointLightProperty, Property, PropertyAddress, PropertyError,
    PropertyKey, PropertyTarget, PropertyValue, RectangleCornersProperty, TrackValue,
    TransformProperty, ViewportProperty, VisibilityProperty,
};
pub use recording::{PropertyWriteFrame, PropertyWriteRecorder, RecordingError, SceneView};
pub use runtime::CompiledTimeline;
pub use time::FrameRange;
pub use track::{Composition, Curve};

#[cfg(test)]
mod tests;
