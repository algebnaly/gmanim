use std::{
    collections::{HashMap, HashSet},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use nalgebra::{Matrix4, Point3, Vector3};

use super::{
    AaLevelProperty, AnimationClip, CameraPoseProperty, CameraProjectionProperty, Curve,
    EnvironmentLightProperty, LayerProperty, PointLightProperty, Property, PropertyAddress,
    PropertyError, RectangleCornersProperty, TransformProperty, ViewportProperty,
    VisibilityProperty,
};
use crate::{
    ClipRect, EnvironmentLight, GMFloat, PointLight, Scene,
    camera::{Camera, CameraPose, Projection},
    mobjects::{MobjectId, SceneWorldError},
};

static NEXT_RECORDING_ID: AtomicU64 = AtomicU64::new(1);

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

trait ErasedRecordProperty: Send + Sync {
    fn address(&self) -> PropertyAddress;
    fn start_track(
        &self,
        baseline: &Scene,
        prefix_samples: u32,
        current: &Scene,
    ) -> Result<Box<dyn RecordedTrack>, PropertyError>;
}

struct RecordProperty<P: Property>(P);

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

trait RecordedTrack: Send {
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

pub struct PropertyWriteFrame {
    recording_id: u64,
    frame: u32,
    duration: u32,
    scene: Scene,
    touched: Vec<Arc<dyn ErasedRecordProperty>>,
    touched_set: HashSet<PropertyAddress>,
}

impl PropertyWriteFrame {
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
            self.touched.push(Arc::new(RecordProperty(property)));
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

pub struct PropertyWriteRecorder {
    id: u64,
    duration: u32,
    next_frame: u32,
    baseline_scene: Scene,
    working_scene: Scene,
    tracks: Vec<Box<dyn RecordedTrack>>,
    track_indices: HashMap<PropertyAddress, usize>,
}

impl PropertyWriteRecorder {
    pub fn new(scene: &Scene, duration: u32) -> Result<Self, RecordingError> {
        if duration == 0 {
            return Err(RecordingError::EmptyRecording);
        }
        Ok(Self {
            id: NEXT_RECORDING_ID.fetch_add(1, Ordering::Relaxed),
            duration,
            next_frame: 1,
            baseline_scene: scene.clone(),
            working_scene: scene.clone(),
            tracks: Vec::new(),
            track_indices: HashMap::new(),
        })
    }

    pub fn duration(&self) -> u32 {
        self.duration
    }

    pub fn next_frame(&self) -> Option<u32> {
        (self.next_frame <= self.duration).then_some(self.next_frame)
    }

    pub fn begin_frame(&self) -> Result<PropertyWriteFrame, RecordingError> {
        if self.next_frame > self.duration {
            return Err(RecordingError::RecordingComplete);
        }
        Ok(PropertyWriteFrame {
            recording_id: self.id,
            frame: self.next_frame,
            duration: self.duration,
            scene: self.working_scene.clone(),
            touched: Vec::new(),
            touched_set: HashSet::new(),
        })
    }

    pub fn commit_frame(&mut self, frame: PropertyWriteFrame) -> Result<(), RecordingError> {
        if frame.recording_id != self.id {
            return Err(RecordingError::ForeignFrame);
        }
        if frame.frame != self.next_frame {
            return Err(RecordingError::UnexpectedFrame {
                expected: self.next_frame,
                actual: frame.frame,
            });
        }

        for track in &mut self.tracks {
            track.push_from(&frame.scene)?;
        }
        for property in frame.touched {
            let address = property.address();
            if self.track_indices.contains_key(&address) {
                continue;
            }
            let track =
                property.start_track(&self.baseline_scene, self.next_frame, &frame.scene)?;
            self.track_indices.insert(address, self.tracks.len());
            self.tracks.push(track);
        }

        self.working_scene = frame.scene;
        self.next_frame += 1;
        Ok(())
    }

    pub fn finish(self) -> Result<AnimationClip, RecordingError> {
        let expected = self.duration + 1;
        if self.next_frame != expected {
            return Err(RecordingError::IncompleteRecording {
                expected,
                actual: self.next_frame,
            });
        }
        let mut clip = AnimationClip::new(self.duration);
        for track in self.tracks {
            clip = track.append_to(clip);
        }
        Ok(clip)
    }
}

#[derive(Debug)]
pub enum RecordingError {
    EmptyRecording,
    RecordingComplete,
    ForeignFrame,
    UnexpectedFrame { expected: u32, actual: u32 },
    IncompleteRecording { expected: u32, actual: u32 },
    Property(PropertyError),
    Scene(SceneWorldError),
}

impl std::fmt::Display for RecordingError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyRecording => {
                formatter.write_str("property recordings must contain at least one frame")
            }
            Self::RecordingComplete => formatter.write_str("all recording frames are complete"),
            Self::ForeignFrame => {
                formatter.write_str("property frame belongs to a different recorder")
            }
            Self::UnexpectedFrame { expected, actual } => {
                write!(
                    formatter,
                    "expected recording frame {expected}, got {actual}"
                )
            }
            Self::IncompleteRecording { expected, actual } => write!(
                formatter,
                "property recording contains {actual} samples; expected {expected}"
            ),
            Self::Property(error) => error.fmt(formatter),
            Self::Scene(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for RecordingError {}

impl From<PropertyError> for RecordingError {
    fn from(error: PropertyError) -> Self {
        Self::Property(error)
    }
}

impl From<SceneWorldError> for RecordingError {
    fn from(error: SceneWorldError) -> Self {
        Self::Scene(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Color, Context,
        animation::{PropertyKey, PropertyTarget, TimelineBuilder},
        camera::PerspectiveSetting,
        mobjects::{Rectangle, SceneWorldError},
    };

    fn assert_send<T: Send>() {}

    #[test]
    fn recording_and_compiled_runtime_are_send() {
        assert_send::<PropertyWriteFrame>();
        assert_send::<PropertyWriteRecorder>();
        assert_send::<AnimationClip>();
        assert_send::<crate::animation::CompiledTimeline>();
    }

    #[test]
    fn scene_view_reads_hierarchy_without_exposing_mutation() {
        let mut scene = Scene::default();
        let parent = scene.world.spawn_group("parent");
        let child = scene.add_rectangle_named("child", Rectangle::default());
        scene.world.set_parent(child, Some(parent)).unwrap();
        scene
            .world
            .get_mut(parent)
            .unwrap()
            .set_transform(Matrix4::new_translation(&Vector3::new(2.0, 0.0, 0.0)));
        scene
            .world
            .get_mut(child)
            .unwrap()
            .set_transform(Matrix4::new_translation(&Vector3::new(3.0, 0.0, 0.0)));

        let view = SceneView::new(&scene);
        assert_eq!(view.find_by_path("parent/child"), Some(child));
        assert_eq!(view.world_transform(child).unwrap()[(0, 3)], 5.0);
        assert_eq!(view.rectangle_corners(child).unwrap().len(), 4);

        scene.world.get_mut(parent).unwrap().set_visible(false);
        assert!(!SceneView::new(&scene).effectively_visible(child).unwrap());
    }

    #[test]
    fn recorder_compiles_owned_frames_to_bidirectional_tracks() {
        let mut scene = Scene::default();
        let rectangle = scene.add_rectangle(Rectangle::default());
        let mut recorder = PropertyWriteRecorder::new(&scene, 3).unwrap();

        for expected_frame in 1..=3 {
            let mut frame = recorder.begin_frame().unwrap();
            assert_eq!(frame.frame(), expected_frame);
            assert_eq!(frame.alpha(), expected_frame as GMFloat / 3.0);
            frame
                .set_position(rectangle, Point3::new(expected_frame as GMFloat, 0.0, 0.0))
                .unwrap();
            assert_eq!(
                frame.view().position(rectangle).unwrap().x,
                expected_frame as GMFloat
            );
            recorder.commit_frame(frame).unwrap();
        }

        let clip = recorder.finish().unwrap();
        let mut builder = TimelineBuilder::new(scene, Context::default());
        builder.append_clip(clip).unwrap();
        let mut timeline = builder.build();

        timeline.seek(3).unwrap();
        assert_eq!(
            timeline.scene.world.get(rectangle).unwrap().transform()[(0, 3)],
            3.0
        );
        timeline.seek(1).unwrap();
        assert_eq!(
            timeline.scene.world.get(rectangle).unwrap().transform()[(0, 3)],
            1.0
        );
    }

    #[test]
    fn recorder_preserves_incremental_callback_semantics() {
        let mut scene = Scene::default();
        let rectangle = scene.add_rectangle(Rectangle::default());
        let mut recorder = PropertyWriteRecorder::new(&scene, 3).unwrap();

        while recorder.next_frame().is_some() {
            let mut frame = recorder.begin_frame().unwrap();
            frame
                .move_by(rectangle, Vector3::new(0.5, 0.0, 0.0))
                .unwrap();
            recorder.commit_frame(frame).unwrap();
        }

        let mut builder = TimelineBuilder::new(scene, Context::default());
        builder.append_clip(recorder.finish().unwrap()).unwrap();
        let mut timeline = builder.build();
        timeline.seek(2).unwrap();
        assert_eq!(
            timeline.scene.world.get(rectangle).unwrap().transform()[(0, 3)],
            1.0
        );
    }

    #[test]
    fn sparse_writes_hold_the_last_committed_value() {
        let mut scene = Scene::default();
        let rectangle = scene.add_rectangle(Rectangle::default());
        let mut recorder = PropertyWriteRecorder::new(&scene, 3).unwrap();

        let frame = recorder.begin_frame().unwrap();
        recorder.commit_frame(frame).unwrap();
        let mut frame = recorder.begin_frame().unwrap();
        frame
            .set_position(rectangle, Point3::new(5.0, 0.0, 0.0))
            .unwrap();
        recorder.commit_frame(frame).unwrap();
        let frame = recorder.begin_frame().unwrap();
        recorder.commit_frame(frame).unwrap();

        let mut builder = TimelineBuilder::new(scene, Context::default());
        builder.append_clip(recorder.finish().unwrap()).unwrap();
        let mut timeline = builder.build();
        timeline.seek(1).unwrap();
        assert_eq!(
            timeline.scene.world.get(rectangle).unwrap().transform()[(0, 3)],
            0.0
        );
        timeline.seek(3).unwrap();
        assert_eq!(
            timeline.scene.world.get(rectangle).unwrap().transform()[(0, 3)],
            5.0
        );
    }

    #[test]
    fn visibility_and_layer_are_typed_tracks() {
        let mut scene = Scene::default();
        let rectangle = scene.add_rectangle(Rectangle::default());
        let mut recorder = PropertyWriteRecorder::new(&scene, 3).unwrap();

        let mut frame = recorder.begin_frame().unwrap();
        frame.set_visible(rectangle, false).unwrap();
        frame.set_layer(rectangle, 4).unwrap();
        recorder.commit_frame(frame).unwrap();
        let frame = recorder.begin_frame().unwrap();
        recorder.commit_frame(frame).unwrap();
        let mut frame = recorder.begin_frame().unwrap();
        frame.set_visible(rectangle, true).unwrap();
        recorder.commit_frame(frame).unwrap();

        let mut builder = TimelineBuilder::new(scene, Context::default());
        builder.append_clip(recorder.finish().unwrap()).unwrap();
        let mut timeline = builder.build();
        timeline.seek(2).unwrap();
        assert!(!timeline.scene.world.get(rectangle).unwrap().visible());
        assert_eq!(timeline.scene.world.get(rectangle).unwrap().layer(), 4);
        timeline.seek(3).unwrap();
        assert!(timeline.scene.world.get(rectangle).unwrap().visible());
        timeline.seek(1).unwrap();
        assert!(!timeline.scene.world.get(rectangle).unwrap().visible());
    }

    #[test]
    fn scene_properties_record_and_seek_together() {
        let scene = Scene::default();
        let mut recorder = PropertyWriteRecorder::new(&scene, 2).unwrap();

        for frame_index in 1..=2 {
            let mut frame = recorder.begin_frame().unwrap();
            frame
                .set_camera_pose(CameraPose {
                    position: Point3::new(frame_index as GMFloat, 2.0, 3.0),
                    look_at: -Vector3::z(),
                    up_direction: Vector3::y(),
                })
                .unwrap();
            frame
                .set_point_light(PointLight {
                    position: Point3::new(0.0, frame_index as GMFloat, 0.0),
                    color: Color::white(),
                    intensity: 100.0 + frame_index as GMFloat,
                })
                .unwrap();
            frame
                .set_environment_light(EnvironmentLight {
                    color: Color::new(10, 20, 30, 255),
                    intensity: frame_index as GMFloat * 0.25,
                    rotation_radians: frame_index as GMFloat,
                })
                .unwrap();
            frame
                .set_camera_projection(Projection::Perspective(PerspectiveSetting::new(
                    16.0 / 9.0,
                    frame_index as GMFloat,
                    0.1,
                    100.0,
                )))
                .unwrap();
            frame
                .set_viewport(Some(ClipRect::Logical(0.0, 0.0, frame_index as f32, 2.0)))
                .unwrap();
            frame.set_aa_level(frame_index * 2).unwrap();
            recorder.commit_frame(frame).unwrap();
        }

        let mut builder = TimelineBuilder::new(scene, Context::default());
        builder.append_clip(recorder.finish().unwrap()).unwrap();
        let mut timeline = builder.build();
        timeline.seek(2).unwrap();
        assert_eq!(timeline.scene.camera.position.x, 2.0);
        assert_eq!(timeline.scene.point_light.position.y, 2.0);
        assert_eq!(timeline.scene.environment_light.intensity, 0.5);
        assert_eq!(timeline.scene.camera.fov(), 2.0);
        assert_eq!(
            timeline.scene.clip_rect,
            Some(ClipRect::Logical(0.0, 0.0, 2.0, 2.0))
        );
        assert_eq!(timeline.scene.aa_level, 4);
        timeline.seek(1).unwrap();
        assert_eq!(timeline.scene.camera.position.x, 1.0);
        assert_eq!(timeline.scene.environment_light.intensity, 0.25);
        assert_eq!(timeline.scene.camera.fov(), 1.0);
        assert_eq!(timeline.scene.aa_level, 2);
    }

    #[derive(Clone, Copy)]
    struct CustomScalarProperty;

    impl Property for CustomScalarProperty {
        type Value = u32;

        fn address(&self) -> PropertyAddress {
            PropertyAddress {
                target: PropertyTarget::Scene,
                key: PropertyKey::new("test", "custom_scalar", "u32"),
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

    #[test]
    fn custom_properties_require_no_timeline_core_changes() {
        let scene = Scene::default();
        let clip = AnimationClip::new(2).track(CustomScalarProperty, Curve::linear(1, 9));
        let mut builder = TimelineBuilder::new(scene, Context::default());
        builder.append_clip(clip).unwrap();
        let mut timeline = builder.build();

        timeline.seek(1).unwrap();
        assert_eq!(timeline.scene.aa_level, 5);
        timeline.seek(2).unwrap();
        assert_eq!(timeline.scene.aa_level, 9);
    }

    #[test]
    fn frames_cannot_cross_recording_sessions() {
        let scene = Scene::default();
        let first = PropertyWriteRecorder::new(&scene, 1).unwrap();
        let mut second = PropertyWriteRecorder::new(&scene, 1).unwrap();

        let error = second
            .commit_frame(first.begin_frame().unwrap())
            .unwrap_err();
        assert!(matches!(error, RecordingError::ForeignFrame));
    }

    #[test]
    fn property_targets_are_validated() {
        let mut scene = Scene::default();
        let group = scene.world.spawn_group("group");
        let recorder = PropertyWriteRecorder::new(&scene, 1).unwrap();
        let mut frame = recorder.begin_frame().unwrap();

        let error = frame
            .set_rectangle_corners(group, [Point3::origin(); 4])
            .unwrap_err();
        assert!(matches!(
            error,
            RecordingError::Property(PropertyError::Scene(
                SceneWorldError::NotRectangle(id)
            )) if id == group
        ));
    }
}
