pub mod basic;
pub mod dot;
pub mod formula;
pub mod mesh_2d;
pub mod mesh_3d;
pub mod object_3d;
pub mod path;
pub mod polygon;
pub mod svg_shape;
pub mod text;
pub mod three_d_viewport;
pub mod wrapper_3d;
pub use basic::{Arc, PolyLine, QuadraticBezier, Rectangle, SimpleLine};
pub use dot::Dot;

use std::{
    borrow::Cow,
    sync::{
        Arc as Shared,
        atomic::{AtomicU64, Ordering},
    },
};

use crate::{Color, Context, GMFloat};
use slotmap::{SecondaryMap, SlotMap, new_key_type};

new_key_type! {
    pub struct MobjectId;
    pub struct RectangleId;
}

static NEXT_WORLD_ID: AtomicU64 = AtomicU64::new(1);

pub enum Geometry3DRef<'a> {
    Mesh(&'a crate::mobjects::mesh_3d::TriangleMesh3D),
    Sdf(&'a dyn crate::mobjects::object_3d::Object3D),
}

pub struct Surface3DSubmission<'a> {
    pub geometry: Geometry3DRef<'a>,
    pub material: crate::mobjects::mesh_3d::SurfaceMaterial,
    pub transform: nalgebra::Matrix4<crate::GMFloat>,
}

pub trait RenderVisitor {
    fn push_mesh_2d(
        &mut self,
        mesh: &crate::mobjects::mesh_2d::TriangleMesh2D,
        transform: nalgebra::Matrix4<crate::GMFloat>,
    );
    fn push_rectangle_2d(
        &mut self,
        id: RectangleId,
        rectangle: &Rectangle,
        geometry_revision: u64,
        dynamic: bool,
        transform: nalgebra::Matrix4<crate::GMFloat>,
    );
    fn push_surface_3d(&mut self, surface: Surface3DSubmission<'_>);
}

pub trait Mobject: Draw + Send + Sync + 'static {
    fn default_name(&self) -> &'static str;

    fn submit_to_renderer(
        &self,
        _visitor: &mut dyn RenderVisitor,
        _world_transform: nalgebra::Matrix4<GMFloat>,
    ) {
    }
}

#[derive(Clone)]
enum VisualComponent {
    None,
    Renderable(Shared<dyn Mobject>),
    Rectangle(RectangleId),
}

#[derive(Clone)]
struct RectangleComponent {
    shape: Rectangle,
    geometry_revision: u64,
    dynamic: bool,
}

#[derive(Clone)]
pub struct MobjectNode {
    name: String,
    parent: Option<MobjectId>,
    children: Vec<MobjectId>,
    transform: nalgebra::Matrix4<GMFloat>,
    visible: bool,
    layer: i32,
    insertion_order: u64,
    visual: VisualComponent,
}

impl std::fmt::Debug for MobjectNode {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MobjectNode")
            .field("name", &self.name)
            .field("parent", &self.parent)
            .field("children", &self.children)
            .field("visible", &self.visible)
            .field("layer", &self.layer)
            .field("insertion_order", &self.insertion_order)
            .finish_non_exhaustive()
    }
}

impl MobjectNode {
    fn new(
        visual: VisualComponent,
        name: String,
        parent: Option<MobjectId>,
        insertion_order: u64,
    ) -> Self {
        Self {
            name,
            parent,
            children: Vec::new(),
            transform: nalgebra::Matrix4::identity(),
            visible: true,
            layer: 0,
            insertion_order,
            visual,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn set_name(&mut self, name: impl Into<String>) {
        self.name = name.into();
    }

    pub fn parent(&self) -> Option<MobjectId> {
        self.parent
    }

    pub fn children(&self) -> &[MobjectId] {
        &self.children
    }

    pub fn transform(&self) -> nalgebra::Matrix4<GMFloat> {
        self.transform
    }

    pub fn set_transform(&mut self, transform: nalgebra::Matrix4<GMFloat>) {
        self.transform = transform;
    }

    pub fn apply_transform(&mut self, transform: nalgebra::Matrix4<GMFloat>) {
        self.transform = transform * self.transform;
    }

    pub fn move_by(&mut self, displacement: nalgebra::Vector3<GMFloat>) {
        self.apply_transform(nalgebra::Matrix4::new_translation(&displacement));
    }

    pub fn scale_by(&mut self, factor: GMFloat) {
        self.apply_transform(nalgebra::Matrix4::new_scaling(factor));
    }

    pub fn visible(&self) -> bool {
        self.visible
    }

    pub fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
    }

    pub fn layer(&self) -> i32 {
        self.layer
    }

    pub fn set_layer(&mut self, layer: i32) {
        self.layer = layer;
    }

    pub fn renderable(&self) -> Option<&dyn Mobject> {
        match &self.visual {
            VisualComponent::Renderable(renderable) => Some(renderable.as_ref()),
            VisualComponent::None | VisualComponent::Rectangle(_) => None,
        }
    }

    pub fn is_group(&self) -> bool {
        matches!(self.visual, VisualComponent::None)
    }

    pub fn is_rectangle(&self) -> bool {
        matches!(self.visual, VisualComponent::Rectangle(_))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SceneWorldError {
    InvalidObjectId(MobjectId),
    ReservedObject(MobjectId),
    ObjectAlreadyLive(MobjectId),
    ForeignSpawnPlan,
    NotRectangle(MobjectId),
    ParentCycle,
}

impl std::fmt::Display for SceneWorldError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidObjectId(id) => write!(formatter, "invalid mobject ID {id:?}"),
            Self::ReservedObject(id) => {
                write!(formatter, "mobject {id:?} is reserved but not live")
            }
            Self::ObjectAlreadyLive(id) => write!(formatter, "mobject {id:?} is already live"),
            Self::ForeignSpawnPlan => {
                formatter.write_str("spawn plan belongs to a different scene world")
            }
            Self::NotRectangle(id) => write!(formatter, "mobject {id:?} is not a rectangle"),
            Self::ParentCycle => formatter.write_str("mobject hierarchy cannot contain a cycle"),
        }
    }
}

impl std::error::Error for SceneWorldError {}

#[derive(Clone)]
pub struct SceneWorld {
    world_id: u64,
    identities: SlotMap<MobjectId, ()>,
    nodes: SecondaryMap<MobjectId, MobjectNode>,
    rectangles: SlotMap<RectangleId, RectangleComponent>,
    roots: Vec<MobjectId>,
    next_insertion_order: u64,
}

impl Default for SceneWorld {
    fn default() -> Self {
        Self {
            world_id: NEXT_WORLD_ID.fetch_add(1, Ordering::Relaxed),
            identities: SlotMap::with_key(),
            nodes: SecondaryMap::new(),
            rectangles: SlotMap::with_key(),
            roots: Vec::new(),
            next_insertion_order: 0,
        }
    }
}

impl SceneWorld {
    pub fn spawn(&mut self, mobject: impl Mobject) -> MobjectId {
        let name = mobject.default_name().to_owned();
        self.spawn_named(name, mobject)
    }

    pub fn spawn_named(&mut self, name: impl Into<String>, mobject: impl Mobject) -> MobjectId {
        self.spawn_tree(NodeBundle::new_named(name, mobject))
    }

    pub fn spawn_rectangle(&mut self, rectangle: Rectangle) -> MobjectId {
        self.spawn_rectangle_named("Rectangle", rectangle)
    }

    pub fn spawn_rectangle_named(
        &mut self,
        name: impl Into<String>,
        rectangle: Rectangle,
    ) -> MobjectId {
        self.spawn_tree(NodeBundle::rectangle_named(name, rectangle))
    }

    pub fn spawn_group(&mut self, name: impl Into<String>) -> MobjectId {
        self.spawn_tree(NodeBundle::group(name))
    }

    pub fn spawn_tree(&mut self, bundle: NodeBundle) -> MobjectId {
        let plan = self.reserve_tree(bundle, None);
        let root = plan.root();
        self.materialize(&plan)
            .expect("newly reserved mobject tree failed to materialize");
        root
    }

    pub fn reserve_tree(&mut self, bundle: NodeBundle, parent: Option<MobjectId>) -> SpawnPlan {
        let root = self.reserve_bundle(bundle);
        SpawnPlan {
            world_id: self.world_id,
            root,
            parent,
        }
    }

    fn reserve_bundle(&mut self, bundle: NodeBundle) -> ReservedNode {
        let id = self.identities.insert(());
        let children = bundle
            .children
            .into_iter()
            .map(|child| self.reserve_bundle(child))
            .collect();
        ReservedNode {
            id,
            name: bundle.name,
            transform: bundle.transform,
            visual: bundle.visual,
            children,
        }
    }

    pub fn materialize(&mut self, plan: &SpawnPlan) -> Result<(), SceneWorldError> {
        if plan.world_id != self.world_id {
            return Err(SceneWorldError::ForeignSpawnPlan);
        }
        if let Some(parent) = plan.parent {
            self.get(parent)?;
        }
        self.validate_reserved_node(&plan.root)?;
        self.materialize_reserved_node(&plan.root, plan.parent)
    }

    fn validate_reserved_node(&self, node: &ReservedNode) -> Result<(), SceneWorldError> {
        if !self.identities.contains_key(node.id) {
            return Err(SceneWorldError::InvalidObjectId(node.id));
        }
        if self.nodes.contains_key(node.id) {
            return Err(SceneWorldError::ObjectAlreadyLive(node.id));
        }
        for child in &node.children {
            self.validate_reserved_node(child)?;
        }
        Ok(())
    }

    fn materialize_reserved_node(
        &mut self,
        reserved: &ReservedNode,
        parent: Option<MobjectId>,
    ) -> Result<(), SceneWorldError> {
        let visual = match &reserved.visual {
            NodeVisual::None => VisualComponent::None,
            NodeVisual::Renderable(renderable) => VisualComponent::Renderable(renderable.clone()),
            NodeVisual::Rectangle(rectangle) => {
                let rectangle = self.rectangles.insert(RectangleComponent {
                    shape: rectangle.clone(),
                    geometry_revision: 0,
                    dynamic: false,
                });
                VisualComponent::Rectangle(rectangle)
            }
        };
        let mut node = MobjectNode::new(
            visual,
            reserved.name.clone(),
            parent,
            self.next_insertion_order,
        );
        node.transform = reserved.transform;
        self.next_insertion_order = self
            .next_insertion_order
            .checked_add(1)
            .expect("mobject insertion order overflowed");
        self.nodes.insert(reserved.id, node);
        match parent {
            Some(parent) => self.get_mut(parent)?.children.push(reserved.id),
            None => self.roots.push(reserved.id),
        }
        for child in &reserved.children {
            self.materialize_reserved_node(child, Some(reserved.id))?;
        }
        Ok(())
    }

    pub(crate) fn synchronize_identities_from(&mut self, other: &SceneWorld) {
        self.identities = other.identities.clone();
    }

    pub fn get(&self, id: MobjectId) -> Result<&MobjectNode, SceneWorldError> {
        if !self.identities.contains_key(id) {
            return Err(SceneWorldError::InvalidObjectId(id));
        }
        self.nodes
            .get(id)
            .ok_or(SceneWorldError::ReservedObject(id))
    }

    pub fn get_mut(&mut self, id: MobjectId) -> Result<&mut MobjectNode, SceneWorldError> {
        if !self.identities.contains_key(id) {
            return Err(SceneWorldError::InvalidObjectId(id));
        }
        self.nodes
            .get_mut(id)
            .ok_or(SceneWorldError::ReservedObject(id))
    }

    pub fn contains(&self, id: MobjectId) -> bool {
        self.nodes.contains_key(id)
    }

    pub fn is_reserved(&self, id: MobjectId) -> bool {
        self.identities.contains_key(id) && !self.nodes.contains_key(id)
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn roots(&self) -> Vec<MobjectId> {
        self.ordered_ids(&self.roots).into_owned()
    }

    pub fn children(&self, id: MobjectId) -> Result<Vec<MobjectId>, SceneWorldError> {
        Ok(self.ordered_ids(&self.get(id)?.children).into_owned())
    }

    pub fn rectangle(&self, id: MobjectId) -> Result<&Rectangle, SceneWorldError> {
        let VisualComponent::Rectangle(rectangle) = self.get(id)?.visual else {
            return Err(SceneWorldError::NotRectangle(id));
        };
        Ok(&self.rectangles[rectangle].shape)
    }

    pub fn rectangle_geometry_revision(&self, id: MobjectId) -> Result<u64, SceneWorldError> {
        let VisualComponent::Rectangle(rectangle) = self.get(id)?.visual else {
            return Err(SceneWorldError::NotRectangle(id));
        };
        Ok(self.rectangles[rectangle].geometry_revision)
    }

    pub fn set_rectangle_corners(
        &mut self,
        id: MobjectId,
        corners: [nalgebra::Point3<GMFloat>; 4],
    ) -> Result<(), SceneWorldError> {
        let VisualComponent::Rectangle(rectangle) = self.get(id)?.visual else {
            return Err(SceneWorldError::NotRectangle(id));
        };
        let component = &mut self.rectangles[rectangle];
        component.shape.set_corners(corners);
        component.geometry_revision = component.geometry_revision.wrapping_add(1);
        component.dynamic = true;
        Ok(())
    }

    pub fn freeze_rectangle_geometry(&mut self, id: MobjectId) -> Result<(), SceneWorldError> {
        let VisualComponent::Rectangle(rectangle) = self.get(id)?.visual else {
            return Err(SceneWorldError::NotRectangle(id));
        };
        self.rectangles[rectangle].dynamic = false;
        Ok(())
    }

    pub fn set_parent(
        &mut self,
        child: MobjectId,
        parent: Option<MobjectId>,
    ) -> Result<(), SceneWorldError> {
        self.get(child)?;
        if let Some(parent) = parent {
            self.get(parent)?;
            let mut ancestor = Some(parent);
            while let Some(id) = ancestor {
                if id == child {
                    return Err(SceneWorldError::ParentCycle);
                }
                ancestor = self.get(id)?.parent;
            }
        }

        let old_parent = self.get(child)?.parent;
        if old_parent == parent {
            return Ok(());
        }
        match old_parent {
            Some(old_parent) => self.get_mut(old_parent)?.children.retain(|id| *id != child),
            None => self.roots.retain(|id| *id != child),
        }
        self.get_mut(child)?.parent = parent;
        match parent {
            Some(parent) => self.get_mut(parent)?.children.push(child),
            None => self.roots.push(child),
        }
        Ok(())
    }

    pub fn remove(&mut self, id: MobjectId) -> Result<Vec<MobjectId>, SceneWorldError> {
        let removed = self.unspawn(id)?;
        for id in &removed {
            self.identities.remove(*id);
        }
        Ok(removed)
    }

    pub fn unspawn(&mut self, id: MobjectId) -> Result<Vec<MobjectId>, SceneWorldError> {
        let parent = self.get(id)?.parent;
        match parent {
            Some(parent) => self.get_mut(parent)?.children.retain(|child| *child != id),
            None => self.roots.retain(|root| *root != id),
        }

        let mut pending = vec![id];
        let mut removed = Vec::new();
        while let Some(current) = pending.pop() {
            let node = self
                .nodes
                .remove(current)
                .ok_or(SceneWorldError::ReservedObject(current))?;
            if let VisualComponent::Rectangle(rectangle) = node.visual {
                self.rectangles.remove(rectangle);
            }
            pending.extend(node.children);
            removed.push(current);
        }
        Ok(removed)
    }

    pub fn find_by_path(&self, path: &str) -> Option<MobjectId> {
        let mut components = path.split('/').filter(|component| !component.is_empty());
        let first = components.next()?;
        let mut current = self
            .roots()
            .into_iter()
            .find(|id| self.nodes[*id].name == first)?;
        for component in components {
            current = self
                .children(current)
                .ok()?
                .into_iter()
                .find(|id| self.nodes[*id].name == component)?;
        }
        Some(current)
    }

    pub fn submit_to_renderer(&self, visitor: &mut dyn RenderVisitor) {
        for root in self.ordered_ids(&self.roots).iter().copied() {
            self.submit_node(root, nalgebra::Matrix4::identity(), true, visitor);
        }
    }

    fn submit_node(
        &self,
        id: MobjectId,
        parent_transform: nalgebra::Matrix4<GMFloat>,
        parent_visible: bool,
        visitor: &mut dyn RenderVisitor,
    ) {
        let node = &self.nodes[id];
        let visible = parent_visible && node.visible;
        if !visible {
            return;
        }
        let world_transform = parent_transform * node.transform;
        match &node.visual {
            VisualComponent::None => {}
            VisualComponent::Renderable(renderable) => {
                renderable.submit_to_renderer(visitor, world_transform);
            }
            VisualComponent::Rectangle(rectangle) => {
                let component = &self.rectangles[*rectangle];
                visitor.push_rectangle_2d(
                    *rectangle,
                    &component.shape,
                    component.geometry_revision,
                    component.dynamic,
                    world_transform,
                );
            }
        }
        for child in self.ordered_ids(&node.children).iter().copied() {
            self.submit_node(child, world_transform, visible, visitor);
        }
    }

    fn ordered_ids<'a>(&self, ids: &'a [MobjectId]) -> Cow<'a, [MobjectId]> {
        let order = |id: MobjectId| {
            let node = &self.nodes[id];
            (node.layer, node.insertion_order)
        };
        if ids.windows(2).all(|pair| order(pair[0]) <= order(pair[1])) {
            return Cow::Borrowed(ids);
        }

        let mut sorted = ids.to_vec();
        sorted.sort_by_key(|id| order(*id));
        Cow::Owned(sorted)
    }
}

#[derive(Clone)]
pub enum NodeVisual {
    None,
    Renderable(Shared<dyn Mobject>),
    Rectangle(Rectangle),
}

#[derive(Clone)]
pub struct NodeBundle {
    pub name: String,
    pub transform: nalgebra::Matrix4<GMFloat>,
    pub visual: NodeVisual,
    pub children: Vec<NodeBundle>,
}

impl NodeBundle {
    pub fn new(mobject: impl Mobject) -> Self {
        let name = mobject.default_name().to_owned();
        Self::new_named(name, mobject)
    }

    pub fn new_named(name: impl Into<String>, mobject: impl Mobject) -> Self {
        Self {
            name: name.into(),
            transform: nalgebra::Matrix4::identity(),
            visual: NodeVisual::Renderable(Shared::new(mobject)),
            children: Vec::new(),
        }
    }

    pub fn group(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            transform: nalgebra::Matrix4::identity(),
            visual: NodeVisual::None,
            children: Vec::new(),
        }
    }

    pub fn rectangle(rectangle: Rectangle) -> Self {
        Self::rectangle_named("Rectangle", rectangle)
    }

    pub fn rectangle_named(name: impl Into<String>, rectangle: Rectangle) -> Self {
        Self {
            name: name.into(),
            transform: nalgebra::Matrix4::identity(),
            visual: NodeVisual::Rectangle(rectangle),
            children: Vec::new(),
        }
    }

    pub fn with_transform(mut self, transform: nalgebra::Matrix4<GMFloat>) -> Self {
        self.transform = transform;
        self
    }

    pub fn with_child(mut self, child: NodeBundle) -> Self {
        self.children.push(child);
        self
    }
}

#[derive(Clone)]
struct ReservedNode {
    id: MobjectId,
    name: String,
    transform: nalgebra::Matrix4<GMFloat>,
    visual: NodeVisual,
    children: Vec<ReservedNode>,
}

#[derive(Clone)]
pub struct SpawnPlan {
    world_id: u64,
    root: ReservedNode,
    parent: Option<MobjectId>,
}

impl SpawnPlan {
    pub fn root(&self) -> MobjectId {
        self.root.id
    }

    pub fn parent(&self) -> Option<MobjectId> {
        self.parent
    }

    pub fn ids(&self) -> Vec<MobjectId> {
        fn collect(node: &ReservedNode, ids: &mut Vec<MobjectId>) {
            ids.push(node.id);
            for child in &node.children {
                collect(child, ids);
            }
        }

        let mut ids = Vec::new();
        collect(&self.root, &mut ids);
        ids
    }
}

impl std::fmt::Debug for SpawnPlan {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SpawnPlan")
            .field("world_id", &self.world_id)
            .field("root", &self.root.id)
            .field("parent", &self.parent)
            .field("ids", &self.ids())
            .finish()
    }
}

pub trait Draw {
    fn draw(&self, ctx: &mut Context, parent_matrix: nalgebra::Matrix4<GMFloat>);
}

#[derive(Debug, Clone, Copy)]
pub struct DrawConfig {
    pub stoke_width: GMFloat,
    pub fill: bool,
    pub color: Color,
}

impl Default for DrawConfig {
    fn default() -> Self {
        DrawConfig {
            stoke_width: 0.25,
            fill: true,
            color: Default::default(),
        }
    }
}

pub fn rotate_matrix(axis: nalgebra::Vector3<GMFloat>, theta: GMFloat) {
    let _ = (axis, theta);
}

#[inline]
pub fn coordinate_change_x(position_x: GMFloat, scene_width: GMFloat) -> GMFloat {
    scene_width / 2.0 + position_x
}

#[inline]
pub fn coordinate_change_y(position_y: GMFloat, scene_height: GMFloat) -> GMFloat {
    scene_height / 2.0 - position_y
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mobjects::mesh_2d::TriangleMesh2D;

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
        let tree = NodeBundle::group("root")
            .with_child(NodeBundle::new(TestMesh(TriangleMesh2D::default())));
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

        let VisualComponent::Renderable(current) = &world.nodes[id].visual else {
            panic!("expected renderable");
        };
        let VisualComponent::Renderable(saved) = &snapshot.nodes[id].visual else {
            panic!("expected renderable");
        };
        assert!(Shared::ptr_eq(current, saved));
        world
            .get_mut(id)
            .unwrap()
            .move_by(nalgebra::Vector3::new(2.0, 0.0, 0.0));
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
            nalgebra::Point3::new(-2.0, 1.0, 0.0),
            nalgebra::Point3::new(-1.0, -1.0, 0.0),
            nalgebra::Point3::new(2.0, -0.5, 0.0),
            nalgebra::Point3::new(1.5, 2.0, 0.0),
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

    impl Draw for TestMesh {
        fn draw(&self, _ctx: &mut Context, _parent_matrix: nalgebra::Matrix4<GMFloat>) {}
    }

    impl Mobject for TestMesh {
        fn default_name(&self) -> &'static str {
            "TestMesh"
        }

        fn submit_to_renderer(
            &self,
            visitor: &mut dyn RenderVisitor,
            world_transform: nalgebra::Matrix4<GMFloat>,
        ) {
            visitor.push_mesh_2d(&self.0, world_transform);
        }
    }

    #[derive(Default)]
    struct TransformRecorder(Vec<nalgebra::Matrix4<GMFloat>>);

    impl RenderVisitor for TransformRecorder {
        fn push_mesh_2d(&mut self, _mesh: &TriangleMesh2D, transform: nalgebra::Matrix4<GMFloat>) {
            self.0.push(transform);
        }

        fn push_rectangle_2d(
            &mut self,
            _id: RectangleId,
            _rectangle: &Rectangle,
            _geometry_revision: u64,
            _dynamic: bool,
            transform: nalgebra::Matrix4<GMFloat>,
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
            .move_by(nalgebra::Vector3::new(2.0, 0.0, 0.0));
        world
            .get_mut(child)
            .unwrap()
            .move_by(nalgebra::Vector3::new(0.0, 3.0, 0.0));

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
}
