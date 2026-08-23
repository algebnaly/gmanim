use nalgebra::{Matrix4, Point3};

use crate::{
    ClipRect, EnvironmentLight, GMFloat, PointLight, Scene,
    camera::{CameraPose, Projection},
    mobjects::{MobjectId, SceneWorldError},
};

use super::protocol::{Property, PropertyAddress, PropertyError, PropertyKey, PropertyTarget};

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
