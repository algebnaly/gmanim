use nalgebra::Matrix4;

use crate::{
    GMFloat, Scene,
    camera::{OrthographicSetting, PerspectiveSetting, Projection},
    mobjects::Rectangle,
};

use super::{
    builtins::TransformProperty,
    protocol::{ErasedProperty, PropertyError, PropertyTarget, PropertyValue, TrackValue},
};

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
