use std::{collections::HashSet, sync::Arc};

use nalgebra::{Matrix4, Point3, Vector3};

use crate::{
    ClipRect, EnvironmentLight, GMFloat, PointLight, Scene,
    camera::{CameraPose, Projection},
    mobjects::MobjectId,
};

use super::{
    super::property::{
        AaLevelProperty, CameraPoseProperty, CameraProjectionProperty, EnvironmentLightProperty,
        LayerProperty, PointLightProperty, Property, PropertyAddress, RectangleCornersProperty,
        TransformProperty, ViewportProperty, VisibilityProperty,
    },
    error::RecordingError,
    recorded_track::{ErasedRecordProperty, RecordProperty},
    scene_view::SceneView,
};

pub struct PropertyWriteFrame {
    recording_id: u64,
    frame: u32,
    duration: u32,
    scene: Scene,
    touched: Vec<Arc<dyn ErasedRecordProperty>>,
    touched_set: HashSet<PropertyAddress>,
}

pub(super) struct RecordedFrame {
    pub(super) recording_id: u64,
    pub(super) frame: u32,
    pub(super) scene: Scene,
    pub(super) touched: Vec<Arc<dyn ErasedRecordProperty>>,
}

impl PropertyWriteFrame {
    pub(super) fn new(recording_id: u64, frame: u32, duration: u32, scene: Scene) -> Self {
        Self {
            recording_id,
            frame,
            duration,
            scene,
            touched: Vec::new(),
            touched_set: HashSet::new(),
        }
    }

    pub(super) fn into_recorded(self) -> RecordedFrame {
        RecordedFrame {
            recording_id: self.recording_id,
            frame: self.frame,
            scene: self.scene,
            touched: self.touched,
        }
    }

    pub fn frame(&self) -> u32 {
        self.frame
    }

    pub fn duration(&self) -> u32 {
        self.duration
    }

    pub fn alpha(&self) -> GMFloat {
        self.frame as GMFloat / self.duration as GMFloat
    }

    pub fn view(&self) -> SceneView<'_> {
        SceneView::new(&self.scene)
    }

    pub fn property<P: Property>(&self, property: &P) -> Result<P::Value, RecordingError> {
        Ok(property.read(&self.scene)?)
    }

    pub fn write<P: Property>(
        &mut self,
        property: P,
        value: P::Value,
    ) -> Result<(), RecordingError> {
        property.write(&mut self.scene, value)?;
        let address = property.address();
        if self.touched_set.insert(address) {
            self.touched.push(Arc::new(RecordProperty::new(property)));
        }
        Ok(())
    }

    pub fn set_transform(
        &mut self,
        target: MobjectId,
        transform: Matrix4<GMFloat>,
    ) -> Result<(), RecordingError> {
        self.write(TransformProperty::new(target), transform)
    }

    pub fn apply_transform(
        &mut self,
        target: MobjectId,
        transform: Matrix4<GMFloat>,
    ) -> Result<(), RecordingError> {
        let current = self.view().transform(target)?;
        self.set_transform(target, transform * current)
    }

    pub fn set_position(
        &mut self,
        target: MobjectId,
        position: Point3<GMFloat>,
    ) -> Result<(), RecordingError> {
        let mut transform = self.view().transform(target)?;
        transform[(0, 3)] = position.x;
        transform[(1, 3)] = position.y;
        transform[(2, 3)] = position.z;
        self.set_transform(target, transform)
    }

    pub fn move_by(
        &mut self,
        target: MobjectId,
        displacement: Vector3<GMFloat>,
    ) -> Result<(), RecordingError> {
        self.apply_transform(target, Matrix4::new_translation(&displacement))
    }

    pub fn set_rectangle_corners(
        &mut self,
        target: MobjectId,
        corners: [Point3<GMFloat>; 4],
    ) -> Result<(), RecordingError> {
        self.write(RectangleCornersProperty::new(target), corners)
    }

    pub fn set_visible(&mut self, target: MobjectId, visible: bool) -> Result<(), RecordingError> {
        self.write(VisibilityProperty::new(target), visible)
    }

    pub fn set_layer(&mut self, target: MobjectId, layer: i32) -> Result<(), RecordingError> {
        self.write(LayerProperty::new(target), layer)
    }

    pub fn set_camera_pose(&mut self, pose: CameraPose) -> Result<(), RecordingError> {
        self.write(CameraPoseProperty, pose)
    }

    pub fn set_camera_projection(&mut self, projection: Projection) -> Result<(), RecordingError> {
        self.write(CameraProjectionProperty, projection)
    }

    pub fn set_point_light(&mut self, light: PointLight) -> Result<(), RecordingError> {
        self.write(PointLightProperty, light)
    }

    pub fn set_environment_light(&mut self, light: EnvironmentLight) -> Result<(), RecordingError> {
        self.write(EnvironmentLightProperty, light)
    }

    pub fn set_viewport(&mut self, viewport: Option<ClipRect>) -> Result<(), RecordingError> {
        self.write(ViewportProperty, viewport)
    }

    pub fn set_aa_level(&mut self, level: u32) -> Result<(), RecordingError> {
        self.write(AaLevelProperty, level)
    }
}
