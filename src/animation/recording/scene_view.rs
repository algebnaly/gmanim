use nalgebra::{Matrix4, Point3};

use crate::{
    ClipRect, EnvironmentLight, GMFloat, PointLight, Scene,
    camera::{Camera, CameraPose, Projection},
    mobjects::{MobjectId, SceneWorldError},
};

use super::super::property::{Property, PropertyError};

#[derive(Clone, Copy)]
pub struct SceneView<'a> {
    scene: &'a Scene,
}

impl<'a> SceneView<'a> {
    pub fn new(scene: &'a Scene) -> Self {
        Self { scene }
    }

    pub fn contains(self, target: MobjectId) -> bool {
        self.scene.world.contains(target)
    }

    pub fn is_reserved(self, target: MobjectId) -> bool {
        self.scene.world.is_reserved(target)
    }

    pub fn roots(self) -> Vec<MobjectId> {
        self.scene.world.roots()
    }

    pub fn children(self, target: MobjectId) -> Result<Vec<MobjectId>, SceneWorldError> {
        self.scene.world.children(target)
    }

    pub fn find_by_path(self, path: &str) -> Option<MobjectId> {
        self.scene.world.find_by_path(path)
    }

    pub fn name(self, target: MobjectId) -> Result<&'a str, SceneWorldError> {
        Ok(self.scene.world.get(target)?.name())
    }

    pub fn parent(self, target: MobjectId) -> Result<Option<MobjectId>, SceneWorldError> {
        Ok(self.scene.world.get(target)?.parent())
    }

    pub fn transform(self, target: MobjectId) -> Result<Matrix4<GMFloat>, SceneWorldError> {
        Ok(self.scene.world.get(target)?.transform())
    }

    pub fn world_transform(self, target: MobjectId) -> Result<Matrix4<GMFloat>, SceneWorldError> {
        let mut chain = Vec::new();
        let mut current = Some(target);
        while let Some(id) = current {
            let node = self.scene.world.get(id)?;
            chain.push(node.transform());
            current = node.parent();
        }
        Ok(chain
            .into_iter()
            .rev()
            .fold(Matrix4::identity(), |parent, local| parent * local))
    }

    pub fn position(self, target: MobjectId) -> Result<Point3<GMFloat>, SceneWorldError> {
        let transform = self.transform(target)?;
        Ok(Point3::new(
            transform[(0, 3)],
            transform[(1, 3)],
            transform[(2, 3)],
        ))
    }

    pub fn visible(self, target: MobjectId) -> Result<bool, SceneWorldError> {
        Ok(self.scene.world.get(target)?.visible())
    }

    pub fn effectively_visible(self, target: MobjectId) -> Result<bool, SceneWorldError> {
        let mut current = Some(target);
        while let Some(id) = current {
            let node = self.scene.world.get(id)?;
            if !node.visible() {
                return Ok(false);
            }
            current = node.parent();
        }
        Ok(true)
    }

    pub fn layer(self, target: MobjectId) -> Result<i32, SceneWorldError> {
        Ok(self.scene.world.get(target)?.layer())
    }

    pub fn is_group(self, target: MobjectId) -> Result<bool, SceneWorldError> {
        Ok(self.scene.world.get(target)?.is_group())
    }

    pub fn is_rectangle(self, target: MobjectId) -> Result<bool, SceneWorldError> {
        Ok(self.scene.world.get(target)?.is_rectangle())
    }

    pub fn rectangle_corners(
        self,
        target: MobjectId,
    ) -> Result<[Point3<GMFloat>; 4], SceneWorldError> {
        Ok(self.scene.world.rectangle(target)?.corners())
    }

    pub fn property<P: Property>(self, property: &P) -> Result<P::Value, PropertyError> {
        property.read(self.scene)
    }

    pub fn camera(self) -> &'a Camera {
        &self.scene.camera
    }

    pub fn camera_pose(self) -> CameraPose {
        self.scene.camera.pose()
    }

    pub fn camera_projection(self) -> &'a Projection {
        self.scene.camera.projection()
    }

    pub fn point_light(self) -> PointLight {
        self.scene.point_light
    }

    pub fn environment_light(self) -> EnvironmentLight {
        self.scene.environment_light
    }

    pub fn viewport(self) -> Option<ClipRect> {
        self.scene.clip_rect
    }

    pub fn aa_level(self) -> u32 {
        self.scene.aa_level
    }
}
