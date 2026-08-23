mod builtins;
mod interpolation;
mod protocol;

pub use builtins::{
    AaLevelProperty, CameraPoseProperty, CameraProjectionProperty, EnvironmentLightProperty,
    LayerProperty, PointLightProperty, RectangleCornersProperty, TransformProperty,
    ViewportProperty, VisibilityProperty,
};
pub use protocol::{
    ErasedProperty, Property, PropertyAddress, PropertyError, PropertyKey, PropertyTarget,
    PropertyValue, TrackValue,
};

#[cfg(test)]
mod tests;
