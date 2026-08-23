use nalgebra::{Matrix4, Point3, Vector3};

use crate::{
    GMFloat,
    mobjects::{
        Mobject, NodeBundle, Rectangle, RectangleId, RenderVisitor, Surface3DSubmission,
        mesh_2d::TriangleMesh2D,
    },
};

use super::{SceneWorld, SceneWorldError};

fn assert_send<T: Send>() {}

#[test]
fn scene_world_is_send() {
    assert_send::<SceneWorld>();
}

#[test]
fn removed_ids_stay_invalid_after_slot_reuse() {
    let mut world = SceneWorld::default();
    let removed = world.spawn_group("removed");
    world.remove(removed).unwrap();
    let replacement = world.spawn_group("replacement");

    assert_ne!(removed, replacement);
    assert_eq!(
        world.get(removed).unwrap_err(),
        SceneWorldError::InvalidObjectId(removed)
    );
}

#[test]
fn reserved_tree_keeps_ids_across_materialization_cycles() {
    let mut world = SceneWorld::default();
    let plan = world.reserve_tree(
        NodeBundle::group("root").with_child(NodeBundle::rectangle(Rectangle::default())),
        None,
    );
    let ids = plan.ids();

    assert_eq!(ids.len(), 2);
    assert!(ids.iter().all(|id| world.is_reserved(*id)));
    assert_eq!(
        world.get(ids[0]).unwrap_err(),
        SceneWorldError::ReservedObject(ids[0])
    );

    world.materialize(&plan).unwrap();
    assert!(ids.iter().all(|id| world.contains(*id)));
    assert_eq!(world.children(ids[0]).unwrap(), [ids[1]]);
    assert!(world.rectangle(ids[1]).is_ok());

    assert_eq!(world.unspawn(ids[0]).unwrap(), ids);
    assert!(ids.iter().all(|id| world.is_reserved(*id)));
    world.materialize(&plan).unwrap();
    assert_eq!(world.children(ids[0]).unwrap(), [ids[1]]);
}

#[test]
fn spawn_plans_cannot_cross_scene_worlds() {
    let mut first = SceneWorld::default();
    let plan = first.reserve_tree(NodeBundle::group("reserved"), None);
    let mut second = SceneWorld::default();
    second.reserve_tree(NodeBundle::group("other"), None);

    assert_eq!(
        second.materialize(&plan),
        Err(SceneWorldError::ForeignSpawnPlan)
    );
}

#[test]
fn reparenting_updates_both_sides_and_rejects_cycles() {
    let mut world = SceneWorld::default();
    let root = world.spawn_group("root");
    let child = world.spawn_group("child");
    let grandchild = world.spawn_group("grandchild");

    world.set_parent(child, Some(root)).unwrap();
    world.set_parent(grandchild, Some(child)).unwrap();
    assert_eq!(world.children(root).unwrap(), [child]);
    assert_eq!(world.get(child).unwrap().parent(), Some(root));
    assert_eq!(
        world.set_parent(root, Some(grandchild)),
        Err(SceneWorldError::ParentCycle)
    );

    world.set_parent(child, None).unwrap();
    assert!(world.children(root).unwrap().is_empty());
    assert_eq!(world.get(child).unwrap().parent(), None);
}

#[test]
fn removing_a_node_recursively_invalidates_descendants() {
    let mut world = SceneWorld::default();
    let root = world.spawn_group("root");
    let child = world.spawn_group("child");
    let grandchild = world.spawn_group("grandchild");
    world.set_parent(child, Some(root)).unwrap();
    world.set_parent(grandchild, Some(child)).unwrap();

    let removed = world.remove(child).unwrap();
    assert_eq!(removed, [child, grandchild]);
    assert!(!world.contains(child));
    assert!(!world.contains(grandchild));
    assert!(world.children(root).unwrap().is_empty());
}

#[test]
fn layer_then_insertion_order_is_deterministic() {
    let mut world = SceneWorld::default();
    let first = world.spawn_group("first");
    let second = world.spawn_group("second");
    let third = world.spawn_group("third");
    world.get_mut(first).unwrap().set_layer(2);
    world.get_mut(second).unwrap().set_layer(-1);
    world.get_mut(third).unwrap().set_layer(2);

    assert_eq!(world.roots(), [second, first, third]);
}

#[test]
fn paths_follow_the_owned_hierarchy() {
    let mut world = SceneWorld::default();
    let root = world.spawn_group("root");
    let child = world.spawn_group("child");
    world.set_parent(child, Some(root)).unwrap();

    assert_eq!(world.find_by_path("root/child"), Some(child));
    assert_eq!(world.find_by_path("child"), None);
}

#[test]
fn node_bundle_represents_renderless_groups_directly() {
    let tree =
        NodeBundle::group("root").with_child(NodeBundle::new(TestMesh(TriangleMesh2D::default())));
    let mut world = SceneWorld::default();
    let root = world.spawn_tree(tree);
    let child = world.children(root).unwrap()[0];

    assert!(world.get(root).unwrap().renderable().is_none());
    assert!(world.get(child).unwrap().renderable().is_some());
    assert_eq!(world.get(child).unwrap().parent(), Some(root));
}

#[test]
fn cloned_worlds_share_renderables_but_not_instance_state() {
    let mut world = SceneWorld::default();
    let id = world.spawn_named("object", TestMesh(TriangleMesh2D::default()));
    let snapshot = world.clone();

    let current = world.get(id).unwrap().renderable().unwrap();
    let saved = snapshot.get(id).unwrap().renderable().unwrap();
    assert!(std::ptr::eq(current, saved));
    world
        .get_mut(id)
        .unwrap()
        .move_by(Vector3::new(2.0, 0.0, 0.0));
    assert_ne!(
        world.get(id).unwrap().transform(),
        snapshot.get(id).unwrap().transform()
    );
}

#[test]
fn rectangle_geometry_is_owned_by_each_world_snapshot() {
    let mut world = SceneWorld::default();
    let id = world.spawn_rectangle(Rectangle::default());
    let snapshot = world.clone();
    let original = snapshot.rectangle(id).unwrap().corners();
    let changed = [
        Point3::new(-2.0, 1.0, 0.0),
        Point3::new(-1.0, -1.0, 0.0),
        Point3::new(2.0, -0.5, 0.0),
        Point3::new(1.5, 2.0, 0.0),
    ];

    world.set_rectangle_corners(id, changed).unwrap();

    assert_eq!(world.rectangle(id).unwrap().corners(), changed);
    assert_eq!(world.rectangle_geometry_revision(id).unwrap(), 1);
    assert_eq!(snapshot.rectangle(id).unwrap().corners(), original);
    assert_eq!(snapshot.rectangle_geometry_revision(id).unwrap(), 0);
}

#[test]
fn removing_rectangle_node_removes_its_typed_component() {
    let mut world = SceneWorld::default();
    let id = world.spawn_rectangle(Rectangle::default());
    assert_eq!(world.rectangles.len(), 1);

    world.remove(id).unwrap();

    assert!(world.rectangles.is_empty());
}

struct TestMesh(TriangleMesh2D);

impl Mobject for TestMesh {
    fn default_name(&self) -> &'static str {
        "TestMesh"
    }

    fn submit_to_renderer(
        &self,
        visitor: &mut dyn RenderVisitor,
        world_transform: Matrix4<GMFloat>,
    ) {
        visitor.push_mesh_2d(&self.0, world_transform);
    }
}

#[derive(Default)]
struct TransformRecorder(Vec<Matrix4<GMFloat>>);

impl RenderVisitor for TransformRecorder {
    fn push_mesh_2d(&mut self, _mesh: &TriangleMesh2D, transform: Matrix4<GMFloat>) {
        self.0.push(transform);
    }

    fn push_rectangle_2d(
        &mut self,
        _id: RectangleId,
        _rectangle: &Rectangle,
        _geometry_revision: u64,
        _dynamic: bool,
        transform: Matrix4<GMFloat>,
    ) {
        self.0.push(transform);
    }

    fn push_surface_3d(&mut self, _surface: Surface3DSubmission<'_>) {}
}

#[test]
fn rendering_composes_hierarchy_transforms_and_visibility() {
    let mut world = SceneWorld::default();
    let parent = world.spawn_group("parent");
    let child = world.spawn_named("child", TestMesh(TriangleMesh2D::default()));
    world.set_parent(child, Some(parent)).unwrap();
    world
        .get_mut(parent)
        .unwrap()
        .move_by(Vector3::new(2.0, 0.0, 0.0));
    world
        .get_mut(child)
        .unwrap()
        .move_by(Vector3::new(0.0, 3.0, 0.0));

    let mut recorder = TransformRecorder::default();
    world.submit_to_renderer(&mut recorder);
    assert_eq!(recorder.0.len(), 1);
    assert_eq!(recorder.0[0][(0, 3)], 2.0);
    assert_eq!(recorder.0[0][(1, 3)], 3.0);

    world.get_mut(parent).unwrap().set_visible(false);
    recorder.0.clear();
    world.submit_to_renderer(&mut recorder);
    assert!(recorder.0.is_empty());
}
