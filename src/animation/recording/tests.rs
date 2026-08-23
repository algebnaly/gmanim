use nalgebra::{Matrix4, Point3, Vector3};

use crate::{
    ClipRect, Color, Context, EnvironmentLight, GMFloat, PointLight, Scene,
    animation::{
        AnimationClip, CompiledTimeline, Curve, Property, PropertyAddress, PropertyError,
        PropertyKey, PropertyTarget, TimelineBuilder,
    },
    camera::{CameraPose, PerspectiveSetting, Projection},
    mobjects::{Rectangle, SceneWorldError},
};

use super::{PropertyWriteFrame, PropertyWriteRecorder, RecordingError, SceneView};

fn assert_send<T: Send>() {}

#[test]
fn recording_and_compiled_runtime_are_send() {
    assert_send::<PropertyWriteFrame>();
    assert_send::<PropertyWriteRecorder>();
    assert_send::<AnimationClip>();
    assert_send::<CompiledTimeline>();
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
        RecordingError::Property(PropertyError::Scene(SceneWorldError::NotRectangle(id)))
            if id == group
    ));
}
