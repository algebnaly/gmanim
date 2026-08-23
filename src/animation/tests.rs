use nalgebra::{Matrix4, Point3, Vector3};

use crate::{Context, Scene, mobjects::Rectangle};

use super::{
    AaLevelProperty, AnimationClip, CameraPoseProperty, Curve, MorphRectangle, Move,
    SceneOperation, TimelineBuilder, TimelineError, Wait,
};

#[test]
fn seek_is_absolute_and_bidirectional() {
    let mut scene = Scene::default();
    let rectangle = scene.add_rectangle(Rectangle::default());
    let mut builder = TimelineBuilder::new(scene, Context::default());
    builder
        .play(Move::new(rectangle, Vector3::new(10.0, 0.0, 0.0), 10))
        .unwrap();
    let mut timeline = builder.build();

    timeline.seek(7).unwrap();
    assert_eq!(
        timeline.scene.world.get(rectangle).unwrap().transform()[(0, 3)],
        7.0
    );
    timeline.seek(2).unwrap();
    assert_eq!(
        timeline.scene.world.get(rectangle).unwrap().transform()[(0, 3)],
        2.0
    );
    timeline.seek(10).unwrap();
    assert_eq!(
        timeline.scene.world.get(rectangle).unwrap().transform()[(0, 3)],
        10.0
    );
}

#[test]
fn sequential_builders_resolve_start_values_at_compile_time() {
    let mut scene = Scene::default();
    let rectangle = scene.add_rectangle(Rectangle::default());
    let mut builder = TimelineBuilder::new(scene, Context::default());
    builder
        .play(Move::new(rectangle, Vector3::new(2.0, 0.0, 0.0), 2))
        .unwrap();
    builder
        .play(Move::new(rectangle, Vector3::new(3.0, 0.0, 0.0), 3))
        .unwrap();
    let mut timeline = builder.build();

    timeline.seek(5).unwrap();
    assert_eq!(
        timeline.scene.world.get(rectangle).unwrap().transform()[(0, 3)],
        5.0
    );
    timeline.seek(3).unwrap();
    assert_eq!(
        timeline.scene.world.get(rectangle).unwrap().transform()[(0, 3)],
        3.0
    );
}

#[test]
fn morph_rectangle_compiles_to_corner_track() {
    let mut scene = Scene::default();
    let rectangle = scene.add_rectangle(Rectangle::default());
    let target = [
        Point3::new(-1.0, 2.0, 0.0),
        Point3::new(-2.0, -1.0, 0.0),
        Point3::new(3.0, -2.0, 0.0),
        Point3::new(2.0, 1.0, 0.0),
    ];
    let mut builder = TimelineBuilder::new(scene, Context::default());
    builder
        .play(MorphRectangle::new(rectangle, target, 10))
        .unwrap();
    let mut timeline = builder.build();

    timeline.seek(5).unwrap();
    let corners = timeline.scene.world.rectangle(rectangle).unwrap().corners();
    let start = Rectangle::default().corners();
    for index in 0..4 {
        assert_eq!(
            corners[index],
            start[index] + (target[index] - start[index]) * 0.5
        );
    }
    timeline.seek(10).unwrap();
    assert_eq!(
        timeline.scene.world.rectangle(rectangle).unwrap().corners(),
        target
    );
}

#[test]
fn sampled_tracks_validate_frame_count() {
    let mut scene = Scene::default();
    let rectangle = scene.add_rectangle(Rectangle::default());
    let curve = Curve::sampled(vec![Matrix4::identity(); 2]);
    let clip = AnimationClip::new(3).transform(rectangle, curve);
    let mut builder = TimelineBuilder::new(scene, Context::default());

    assert!(matches!(
        builder.play(clip),
        Err(TimelineError::SampleCount {
            expected: 4,
            actual: 2
        })
    ));
}

#[test]
fn duplicate_property_writes_in_one_clip_are_rejected() {
    let mut scene = Scene::default();
    let rectangle = scene.add_rectangle(Rectangle::default());
    let curve = Curve::linear(Matrix4::identity(), Matrix4::new_scaling(2.0));
    let clip = AnimationClip::new(10)
        .transform(rectangle, curve.clone())
        .transform(rectangle, curve);
    let mut builder = TimelineBuilder::new(scene, Context::default());

    assert!(matches!(
        builder.play(clip),
        Err(TimelineError::ConflictingWrites { .. })
    ));
}

#[test]
fn structural_events_replay_after_backward_seek() {
    let mut scene = Scene::default();
    let parent = scene.world.spawn_group("parent");
    let rectangle = scene.add_rectangle(Rectangle::default());
    let clip = AnimationClip::new(10).event(
        5,
        SceneOperation::SetParent {
            child: rectangle,
            parent: Some(parent),
        },
    );
    let mut builder = TimelineBuilder::new(scene, Context::default());
    builder.play(clip).unwrap();
    let mut timeline = builder.build();

    timeline.seek(8).unwrap();
    assert_eq!(
        timeline.scene.world.get(rectangle).unwrap().parent(),
        Some(parent)
    );
    timeline.seek(2).unwrap();
    assert_eq!(timeline.scene.world.get(rectangle).unwrap().parent(), None);
}

#[test]
fn spawn_events_materialize_reserved_ids_at_the_target_frame() {
    let scene = Scene::default();
    let mut builder = TimelineBuilder::new(scene, Context::default());
    let plan = builder
        .reserve_spawn(
            crate::mobjects::NodeBundle::rectangle(Rectangle::default()),
            None,
        )
        .unwrap();
    let rectangle = plan.root();
    builder
        .append_clip(AnimationClip::new(5).spawn(3, plan))
        .unwrap();
    let mut timeline = builder.build();

    assert!(timeline.scene.world.is_reserved(rectangle));
    timeline.seek(2).unwrap();
    assert!(timeline.scene.world.is_reserved(rectangle));
    timeline.seek(3).unwrap();
    assert!(timeline.scene.world.contains(rectangle));
    assert!(timeline.scene.world.rectangle(rectangle).is_ok());
    timeline.seek(1).unwrap();
    assert!(timeline.scene.world.is_reserved(rectangle));
    timeline.seek(5).unwrap();
    assert!(timeline.scene.world.contains(rectangle));
}

#[test]
fn spawned_trees_keep_reserved_child_ids_and_external_parent() {
    use crate::mobjects::NodeBundle;

    let mut scene = Scene::default();
    let parent = scene.world.spawn_group("parent");
    let mut builder = TimelineBuilder::new(scene, Context::default());
    let plan = builder
        .reserve_spawn(
            NodeBundle::group("spawned-root")
                .with_child(NodeBundle::rectangle(Rectangle::default())),
            Some(parent),
        )
        .unwrap();
    let ids = plan.ids();
    builder
        .append_clip(AnimationClip::new(2).spawn(1, plan))
        .unwrap();
    let mut timeline = builder.build();

    timeline.seek(1).unwrap();
    assert_eq!(timeline.scene.world.children(parent).unwrap(), [ids[0]]);
    assert_eq!(timeline.scene.world.children(ids[0]).unwrap(), [ids[1]]);
    assert_eq!(
        timeline.scene.world.get(ids[1]).unwrap().parent(),
        Some(ids[0])
    );
}

#[test]
fn remove_events_preserve_identity_for_backward_seek() {
    use crate::mobjects::NodeBundle;

    let scene = Scene::default();
    let mut builder = TimelineBuilder::new(scene, Context::default());
    let plan = builder
        .reserve_spawn(NodeBundle::rectangle(Rectangle::default()), None)
        .unwrap();
    let rectangle = plan.root();
    builder
        .append_clip(AnimationClip::new(2).spawn(1, plan))
        .unwrap();
    builder
        .append_clip(AnimationClip::new(2).event(1, SceneOperation::Remove { target: rectangle }))
        .unwrap();
    let mut timeline = builder.build();

    timeline.seek(4).unwrap();
    assert!(timeline.scene.world.is_reserved(rectangle));
    timeline.seek(2).unwrap();
    assert!(timeline.scene.world.contains(rectangle));
    timeline.seek(0).unwrap();
    assert!(timeline.scene.world.is_reserved(rectangle));
    timeline.seek(2).unwrap();
    assert!(timeline.scene.world.contains(rectangle));
}

#[test]
fn frame_zero_boundary_operations_fold_into_the_initial_snapshot() {
    use crate::mobjects::NodeBundle;

    let scene = Scene::default();
    let mut builder = TimelineBuilder::new(scene, Context::default());
    let rectangle = builder
        .add(NodeBundle::rectangle(Rectangle::default()), None)
        .unwrap();
    builder.play(Wait::new(2)).unwrap();
    let mut timeline = builder.build();

    assert_eq!(timeline.total_frames(), 2);
    assert!(timeline.scene.world.contains(rectangle));
    timeline.seek(2).unwrap();
    timeline.seek(0).unwrap();
    assert!(timeline.scene.world.contains(rectangle));
}

#[test]
fn later_boundary_operations_do_not_consume_frames() {
    use crate::mobjects::NodeBundle;

    let mut scene = Scene::default();
    let original = scene.add_rectangle(Rectangle::default());
    let mut builder = TimelineBuilder::new(scene, Context::default());
    builder.play(Wait::new(2)).unwrap();
    let added = builder
        .add(NodeBundle::rectangle(Rectangle::default()), None)
        .unwrap();
    builder.remove(original).unwrap();
    builder.play(Wait::new(2)).unwrap();
    let mut timeline = builder.build();

    assert_eq!(timeline.total_frames(), 4);
    timeline.seek(1).unwrap();
    assert!(timeline.scene.world.contains(original));
    assert!(timeline.scene.world.is_reserved(added));
    timeline.seek(2).unwrap();
    assert!(timeline.scene.world.is_reserved(original));
    assert!(timeline.scene.world.contains(added));
    timeline.seek(4).unwrap();
    timeline.seek(1).unwrap();
    assert!(timeline.scene.world.contains(original));
    assert!(timeline.scene.world.is_reserved(added));
}

#[test]
fn scene_properties_can_change_at_timeline_boundaries() {
    let scene = Scene::default();
    let initial_pose = scene.camera.pose();
    let target_pose = crate::camera::CameraPose {
        position: Point3::new(3.0, 4.0, 5.0),
        look_at: -Vector3::z(),
        up_direction: Vector3::y(),
    };
    let mut builder = TimelineBuilder::new(scene, Context::default());
    builder.play(Wait::new(2)).unwrap();
    builder.set(CameraPoseProperty, target_pose).unwrap();
    builder.set(AaLevelProperty, 6).unwrap();
    builder.play(Wait::new(2)).unwrap();
    let mut timeline = builder.build();

    timeline.seek(1).unwrap();
    assert_eq!(timeline.scene.camera.pose(), initial_pose);
    assert_eq!(timeline.scene.aa_level, 1);
    timeline.seek(2).unwrap();
    assert_eq!(timeline.scene.camera.pose(), target_pose);
    assert_eq!(timeline.scene.aa_level, 6);
    timeline.seek(4).unwrap();
    timeline.seek(1).unwrap();
    assert_eq!(timeline.scene.camera.pose(), initial_pose);
    assert_eq!(timeline.scene.aa_level, 1);
}
