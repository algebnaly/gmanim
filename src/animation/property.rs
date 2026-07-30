use std::{any::Any, fmt, sync::Arc};

use nalgebra::{Matrix4, Point3, Vector3};

use crate::{
    ClipRect, Color, EnvironmentLight, GMFloat, PointLight, Scene,
    camera::{CameraPose, OrthographicSetting, PerspectiveSetting, Projection},
    mobjects::{MobjectId, SceneWorldError},
};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PropertyTarget {
    Scene,
    Mobject(MobjectId),
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct PropertyKey {
    pub namespace: &'static str,
    pub name: &'static str,
    pub value_type: &'static str,
}

impl PropertyKey {
    pub const fn new(
        namespace: &'static str,
        name: &'static str,
        value_type: &'static str,
    ) -> Self {
        Self {
            namespace,
            name,
            value_type,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct PropertyAddress {
    pub target: PropertyTarget,
    pub key: PropertyKey,
}

impl fmt::Display for PropertyAddress {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.target {
            PropertyTarget::Scene => {
                write!(formatter, "scene.{}::{}", self.key.namespace, self.key.name)
            }
            PropertyTarget::Mobject(target) => write!(
                formatter,
                "{target:?}.{}::{}",
                self.key.namespace, self.key.name
            ),
        }
    }
}

pub trait TrackValue: Clone + Send + Sync + 'static {
    fn interpolate(from: &Self, to: &Self, alpha: GMFloat) -> Self;
}

pub trait Property: Clone + Send + Sync + 'static {
    type Value: TrackValue;

    fn address(&self) -> PropertyAddress;
    fn read(&self, scene: &Scene) -> Result<Self::Value, PropertyError>;
    fn write(&self, scene: &mut Scene, value: Self::Value) -> Result<(), PropertyError>;

    fn is_present(&self, _scene: &Scene) -> Result<bool, PropertyError> {
        Ok(true)
    }

    fn finalize(&self, _scene: &mut Scene) -> Result<(), PropertyError> {
        Ok(())
    }
}

#[derive(Clone)]
pub struct PropertyValue {
    value_type: &'static str,
    inner: Arc<dyn Any + Send + Sync>,
}

impl PropertyValue {
    pub fn new<T: Clone + Send + Sync + 'static>(value: T) -> Self {
        Self {
            value_type: std::any::type_name::<T>(),
            inner: Arc::new(value),
        }
    }

    pub(crate) fn with_type<T: Clone + Send + Sync + 'static>(
        value_type: &'static str,
        value: T,
    ) -> Self {
        Self {
            value_type,
            inner: Arc::new(value),
        }
    }

    pub fn value_type(&self) -> &'static str {
        self.value_type
    }

    pub fn downcast_ref<T: Send + Sync + 'static>(&self) -> Option<&T> {
        self.inner.downcast_ref()
    }
}

impl fmt::Debug for PropertyValue {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PropertyValue")
            .field("value_type", &self.value_type)
            .finish_non_exhaustive()
    }
}

trait ErasedPropertyBehavior: Send + Sync {
    fn address(&self) -> PropertyAddress;
    fn read(&self, scene: &Scene) -> Result<PropertyValue, PropertyError>;
    fn write(&self, scene: &mut Scene, value: &PropertyValue) -> Result<(), PropertyError>;
}

struct TypedPropertyBehavior<P: Property>(P);

impl<P: Property> ErasedPropertyBehavior for TypedPropertyBehavior<P> {
    fn address(&self) -> PropertyAddress {
        self.0.address()
    }

    fn read(&self, scene: &Scene) -> Result<PropertyValue, PropertyError> {
        Ok(PropertyValue::with_type(
            self.0.address().key.value_type,
            self.0.read(scene)?,
        ))
    }

    fn write(&self, scene: &mut Scene, value: &PropertyValue) -> Result<(), PropertyError> {
        let Some(value) = value.downcast_ref::<P::Value>() else {
            return Err(PropertyError::TypeMismatch {
                address: self.0.address(),
                expected: self.0.address().key.value_type,
                actual: value.value_type(),
            });
        };
        self.0.write(scene, value.clone())
    }
}

#[derive(Clone)]
pub struct ErasedProperty {
    behavior: Arc<dyn ErasedPropertyBehavior>,
}

impl ErasedProperty {
    pub fn new<P: Property>(property: P) -> Self {
        Self {
            behavior: Arc::new(TypedPropertyBehavior(property)),
        }
    }

    pub fn address(&self) -> PropertyAddress {
        self.behavior.address()
    }

    pub fn read(&self, scene: &Scene) -> Result<PropertyValue, PropertyError> {
        self.behavior.read(scene)
    }

    pub fn write(&self, scene: &mut Scene, value: &PropertyValue) -> Result<(), PropertyError> {
        self.behavior.write(scene, value)
    }
}

impl fmt::Debug for ErasedProperty {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("ErasedProperty")
            .field(&self.address())
            .finish()
    }
}

#[derive(Debug)]
pub enum PropertyError {
    TypeMismatch {
        address: PropertyAddress,
        expected: &'static str,
        actual: &'static str,
    },
    Scene(SceneWorldError),
}

impl fmt::Display for PropertyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TypeMismatch {
                address,
                expected,
                actual,
            } => write!(
                formatter,
                "cannot write {actual} to {address}; expected {expected}"
            ),
            Self::Scene(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for PropertyError {}

impl From<SceneWorldError> for PropertyError {
    fn from(error: SceneWorldError) -> Self {
        Self::Scene(error)
    }
}

fn object_is_present(scene: &Scene, target: MobjectId) -> Result<bool, PropertyError> {
    if scene.world.contains(target) {
        return Ok(true);
    }
    if scene.world.is_reserved(target) {
        return Ok(false);
    }
    Err(SceneWorldError::InvalidObjectId(target).into())
}

#[derive(Clone, Copy, Debug)]
pub struct TransformProperty {
    pub target: MobjectId,
}

impl TransformProperty {
    pub const fn new(target: MobjectId) -> Self {
        Self { target }
    }
}

impl Property for TransformProperty {
    type Value = Matrix4<GMFloat>;

    fn address(&self) -> PropertyAddress {
        PropertyAddress {
            target: PropertyTarget::Mobject(self.target),
            key: PropertyKey::new("node", "transform", "mat4"),
        }
    }

    fn read(&self, scene: &Scene) -> Result<Self::Value, PropertyError> {
        Ok(scene.world.get(self.target)?.transform())
    }

    fn write(&self, scene: &mut Scene, value: Self::Value) -> Result<(), PropertyError> {
        scene.world.get_mut(self.target)?.set_transform(value);
        Ok(())
    }

    fn is_present(&self, scene: &Scene) -> Result<bool, PropertyError> {
        object_is_present(scene, self.target)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct VisibilityProperty {
    pub target: MobjectId,
}

impl VisibilityProperty {
    pub const fn new(target: MobjectId) -> Self {
        Self { target }
    }
}

impl Property for VisibilityProperty {
    type Value = bool;

    fn address(&self) -> PropertyAddress {
        PropertyAddress {
            target: PropertyTarget::Mobject(self.target),
            key: PropertyKey::new("node", "visibility", "bool"),
        }
    }

    fn read(&self, scene: &Scene) -> Result<Self::Value, PropertyError> {
        Ok(scene.world.get(self.target)?.visible())
    }

    fn write(&self, scene: &mut Scene, value: Self::Value) -> Result<(), PropertyError> {
        scene.world.get_mut(self.target)?.set_visible(value);
        Ok(())
    }

    fn is_present(&self, scene: &Scene) -> Result<bool, PropertyError> {
        object_is_present(scene, self.target)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LayerProperty {
    pub target: MobjectId,
}

impl LayerProperty {
    pub const fn new(target: MobjectId) -> Self {
        Self { target }
    }
}

impl Property for LayerProperty {
    type Value = i32;

    fn address(&self) -> PropertyAddress {
        PropertyAddress {
            target: PropertyTarget::Mobject(self.target),
            key: PropertyKey::new("node", "layer", "i32"),
        }
    }

    fn read(&self, scene: &Scene) -> Result<Self::Value, PropertyError> {
        Ok(scene.world.get(self.target)?.layer())
    }

    fn write(&self, scene: &mut Scene, value: Self::Value) -> Result<(), PropertyError> {
        scene.world.get_mut(self.target)?.set_layer(value);
        Ok(())
    }

    fn is_present(&self, scene: &Scene) -> Result<bool, PropertyError> {
        object_is_present(scene, self.target)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RectangleCornersProperty {
    pub target: MobjectId,
}

impl RectangleCornersProperty {
    pub const fn new(target: MobjectId) -> Self {
        Self { target }
    }
}

impl Property for RectangleCornersProperty {
    type Value = [Point3<GMFloat>; 4];

    fn address(&self) -> PropertyAddress {
        PropertyAddress {
            target: PropertyTarget::Mobject(self.target),
            key: PropertyKey::new("rectangle", "corners", "point3[4]"),
        }
    }

    fn read(&self, scene: &Scene) -> Result<Self::Value, PropertyError> {
        Ok(scene.world.rectangle(self.target)?.corners())
    }

    fn write(&self, scene: &mut Scene, value: Self::Value) -> Result<(), PropertyError> {
        scene.world.set_rectangle_corners(self.target, value)?;
        Ok(())
    }

    fn is_present(&self, scene: &Scene) -> Result<bool, PropertyError> {
        object_is_present(scene, self.target)
    }

    fn finalize(&self, scene: &mut Scene) -> Result<(), PropertyError> {
        scene.world.freeze_rectangle_geometry(self.target)?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CameraPoseProperty;

impl Property for CameraPoseProperty {
    type Value = CameraPose;

    fn address(&self) -> PropertyAddress {
        PropertyAddress {
            target: PropertyTarget::Scene,
            key: PropertyKey::new("camera", "pose", "camera_pose"),
        }
    }

    fn read(&self, scene: &Scene) -> Result<Self::Value, PropertyError> {
        Ok(scene.camera.pose())
    }

    fn write(&self, scene: &mut Scene, value: Self::Value) -> Result<(), PropertyError> {
        scene.camera.set_pose(value);
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CameraProjectionProperty;

impl Property for CameraProjectionProperty {
    type Value = Projection;

    fn address(&self) -> PropertyAddress {
        PropertyAddress {
            target: PropertyTarget::Scene,
            key: PropertyKey::new("camera", "projection", "projection"),
        }
    }

    fn read(&self, scene: &Scene) -> Result<Self::Value, PropertyError> {
        Ok(scene.camera.projection().clone())
    }

    fn write(&self, scene: &mut Scene, value: Self::Value) -> Result<(), PropertyError> {
        scene.camera.set_projection(value);
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PointLightProperty;

impl Property for PointLightProperty {
    type Value = PointLight;

    fn address(&self) -> PropertyAddress {
        PropertyAddress {
            target: PropertyTarget::Scene,
            key: PropertyKey::new("lighting", "point", "point_light"),
        }
    }

    fn read(&self, scene: &Scene) -> Result<Self::Value, PropertyError> {
        Ok(scene.point_light)
    }

    fn write(&self, scene: &mut Scene, value: Self::Value) -> Result<(), PropertyError> {
        scene.point_light = value;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct EnvironmentLightProperty;

impl Property for EnvironmentLightProperty {
    type Value = EnvironmentLight;

    fn address(&self) -> PropertyAddress {
        PropertyAddress {
            target: PropertyTarget::Scene,
            key: PropertyKey::new("lighting", "environment", "environment_light"),
        }
    }

    fn read(&self, scene: &Scene) -> Result<Self::Value, PropertyError> {
        Ok(scene.environment_light)
    }

    fn write(&self, scene: &mut Scene, value: Self::Value) -> Result<(), PropertyError> {
        scene.environment_light = value;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ViewportProperty;

impl Property for ViewportProperty {
    type Value = Option<ClipRect>;

    fn address(&self) -> PropertyAddress {
        PropertyAddress {
            target: PropertyTarget::Scene,
            key: PropertyKey::new("scene", "viewport", "clip_rect?"),
        }
    }

    fn read(&self, scene: &Scene) -> Result<Self::Value, PropertyError> {
        Ok(scene.clip_rect)
    }

    fn write(&self, scene: &mut Scene, value: Self::Value) -> Result<(), PropertyError> {
        scene.clip_rect = value;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AaLevelProperty;

impl Property for AaLevelProperty {
    type Value = u32;

    fn address(&self) -> PropertyAddress {
        PropertyAddress {
            target: PropertyTarget::Scene,
            key: PropertyKey::new("scene", "aa_level", "u32"),
        }
    }

    fn read(&self, scene: &Scene) -> Result<Self::Value, PropertyError> {
        Ok(scene.aa_level)
    }

    fn write(&self, scene: &mut Scene, value: Self::Value) -> Result<(), PropertyError> {
        scene.aa_level = value;
        Ok(())
    }
}

impl TrackValue for Matrix4<GMFloat> {
    fn interpolate(from: &Self, to: &Self, alpha: GMFloat) -> Self {
        from * (1.0 - alpha) + to * alpha
    }
}

impl TrackValue for [Point3<GMFloat>; 4] {
    fn interpolate(from: &Self, to: &Self, alpha: GMFloat) -> Self {
        std::array::from_fn(|index| from[index] + (to[index] - from[index]) * alpha)
    }
}

impl TrackValue for bool {
    fn interpolate(from: &Self, to: &Self, alpha: GMFloat) -> Self {
        if alpha < 1.0 { *from } else { *to }
    }
}

impl TrackValue for i32 {
    fn interpolate(from: &Self, to: &Self, alpha: GMFloat) -> Self {
        ((*from as GMFloat) * (1.0 - alpha) + (*to as GMFloat) * alpha).round() as i32
    }
}

impl TrackValue for u32 {
    fn interpolate(from: &Self, to: &Self, alpha: GMFloat) -> Self {
        ((*from as GMFloat) * (1.0 - alpha) + (*to as GMFloat) * alpha).round() as u32
    }
}

impl TrackValue for CameraPose {
    fn interpolate(from: &Self, to: &Self, alpha: GMFloat) -> Self {
        Self {
            position: from.position + (to.position - from.position) * alpha,
            look_at: normalized_lerp(from.look_at, to.look_at, alpha),
            up_direction: normalized_lerp(from.up_direction, to.up_direction, alpha),
        }
    }
}

impl TrackValue for Projection {
    fn interpolate(from: &Self, to: &Self, alpha: GMFloat) -> Self {
        match (from, to) {
            (Self::Perspective(from), Self::Perspective(to)) => {
                let (fa, ff, fnr, ffr) = from.params();
                let (ta, tf, tnr, tfr) = to.params();
                Self::Perspective(PerspectiveSetting::new(
                    lerp(fa, ta, alpha),
                    lerp(ff, tf, alpha),
                    lerp(fnr, tnr, alpha),
                    lerp(ffr, tfr, alpha),
                ))
            }
            (Self::Orthographic(from), Self::Orthographic(to)) => {
                let (fl, fr, fb, ft, fnr, ffr) = from.params();
                let (tl, tr, tb, tt, tnr, tfr) = to.params();
                Self::Orthographic(OrthographicSetting::new(
                    lerp(fl, tl, alpha),
                    lerp(fr, tr, alpha),
                    lerp(fb, tb, alpha),
                    lerp(ft, tt, alpha),
                    lerp(fnr, tnr, alpha),
                    lerp(ffr, tfr, alpha),
                ))
            }
            _ if alpha < 1.0 => from.clone(),
            _ => to.clone(),
        }
    }
}

impl TrackValue for PointLight {
    fn interpolate(from: &Self, to: &Self, alpha: GMFloat) -> Self {
        Self {
            position: from.position + (to.position - from.position) * alpha,
            color: interpolate_color(from.color, to.color, alpha),
            intensity: lerp(from.intensity, to.intensity, alpha),
        }
    }
}

impl TrackValue for EnvironmentLight {
    fn interpolate(from: &Self, to: &Self, alpha: GMFloat) -> Self {
        Self {
            color: interpolate_color(from.color, to.color, alpha),
            intensity: lerp(from.intensity, to.intensity, alpha),
            rotation_radians: lerp(from.rotation_radians, to.rotation_radians, alpha),
        }
    }
}

impl TrackValue for Option<ClipRect> {
    fn interpolate(from: &Self, to: &Self, alpha: GMFloat) -> Self {
        match (from, to) {
            (Some(ClipRect::Logical(fx, fy, fw, fh)), Some(ClipRect::Logical(tx, ty, tw, th))) => {
                Some(ClipRect::Logical(
                    lerp_f32(*fx, *tx, alpha),
                    lerp_f32(*fy, *ty, alpha),
                    lerp_f32(*fw, *tw, alpha),
                    lerp_f32(*fh, *th, alpha),
                ))
            }
            (Some(ClipRect::Pixel(fx, fy, fw, fh)), Some(ClipRect::Pixel(tx, ty, tw, th))) => {
                Some(ClipRect::Pixel(
                    lerp_u32(*fx, *tx, alpha),
                    lerp_u32(*fy, *ty, alpha),
                    lerp_u32(*fw, *tw, alpha),
                    lerp_u32(*fh, *th, alpha),
                ))
            }
            _ if alpha < 1.0 => *from,
            _ => *to,
        }
    }
}

fn lerp(from: GMFloat, to: GMFloat, alpha: GMFloat) -> GMFloat {
    from * (1.0 - alpha) + to * alpha
}

fn lerp_f32(from: f32, to: f32, alpha: GMFloat) -> f32 {
    from * (1.0 - alpha) + to * alpha
}

fn lerp_u32(from: u32, to: u32, alpha: GMFloat) -> u32 {
    (from as f64 * (1.0 - alpha as f64) + to as f64 * alpha as f64).round() as u32
}

fn normalized_lerp(
    from: Vector3<GMFloat>,
    to: Vector3<GMFloat>,
    alpha: GMFloat,
) -> Vector3<GMFloat> {
    let value = from * (1.0 - alpha) + to * alpha;
    if value.norm_squared() > GMFloat::EPSILON {
        value.normalize()
    } else {
        to
    }
}

fn interpolate_color(from: Color, to: Color, alpha: GMFloat) -> Color {
    let channel = |from: u8, to: u8| {
        (from as GMFloat * (1.0 - alpha) + to as GMFloat * alpha)
            .round()
            .clamp(0.0, 255.0) as u8
    };
    Color::new(
        channel(from.r, to.r),
        channel(from.g, to.g),
        channel(from.b, to.b),
        channel(from.a, to.a),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mobjects::Rectangle;

    #[test]
    fn erased_properties_preserve_address_and_validate_value_type() {
        let mut scene = Scene::default();
        let rectangle = scene.add_rectangle(Rectangle::default());
        let property = ErasedProperty::new(TransformProperty::new(rectangle));

        assert_eq!(
            property.address().target,
            PropertyTarget::Mobject(rectangle)
        );
        assert!(
            property
                .read(&scene)
                .unwrap()
                .downcast_ref::<Matrix4<GMFloat>>()
                .is_some()
        );
        let error = property
            .write(&mut scene, &PropertyValue::new(false))
            .unwrap_err();
        assert!(matches!(error, PropertyError::TypeMismatch { .. }));
    }

    #[test]
    fn projection_interpolation_handles_matching_and_different_modes() {
        let from = Projection::Perspective(PerspectiveSetting::new(1.0, 1.0, 0.1, 100.0));
        let to = Projection::Perspective(PerspectiveSetting::new(2.0, 2.0, 0.3, 300.0));
        let Projection::Perspective(midpoint) =
            <Projection as TrackValue>::interpolate(&from, &to, 0.5)
        else {
            panic!("projection mode changed unexpectedly");
        };
        assert_eq!(midpoint.params(), (1.5, 1.5, 0.2, 200.0));

        let orthographic =
            Projection::Orthographic(OrthographicSetting::new(-2.0, 2.0, -1.0, 1.0, 0.1, 10.0));
        assert!(matches!(
            <Projection as TrackValue>::interpolate(&from, &orthographic, 0.5),
            Projection::Perspective(_)
        ));
        assert!(matches!(
            <Projection as TrackValue>::interpolate(&from, &orthographic, 1.0),
            Projection::Orthographic(_)
        ));
    }
}
