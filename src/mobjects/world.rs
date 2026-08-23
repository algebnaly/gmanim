use std::{
    borrow::Cow,
    sync::atomic::{AtomicU64, Ordering},
};

use nalgebra::{Matrix4, Point3};
use slotmap::{SecondaryMap, SlotMap};

use crate::GMFloat;

use super::{
    basic::Rectangle,
    mobject::Mobject,
    node::{MobjectId, MobjectNode, RectangleId, VisualComponent},
    spawn::{NodeBundle, NodeVisual, ReservedNode, SpawnPlan},
    submission::RenderVisitor,
};

static NEXT_WORLD_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone)]
struct RectangleComponent {
    shape: Rectangle,
    geometry_revision: u64,
    dynamic: bool,
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
        let identities = &mut self.identities;
        let root = ReservedNode::from_bundle(bundle, &mut || identities.insert(()));
        SpawnPlan::new(self.world_id, root, parent)
    }

    pub fn materialize(&mut self, plan: &SpawnPlan) -> Result<(), SceneWorldError> {
        if plan.world_id() != self.world_id {
            return Err(SceneWorldError::ForeignSpawnPlan);
        }
        if let Some(parent) = plan.parent() {
            self.get(parent)?;
        }
        self.validate_reserved_node(plan.reserved_root())?;
        self.materialize_reserved_node(plan.reserved_root(), plan.parent())
    }

    fn validate_reserved_node(&self, node: &ReservedNode) -> Result<(), SceneWorldError> {
        if !self.identities.contains_key(node.id()) {
            return Err(SceneWorldError::InvalidObjectId(node.id()));
        }
        if self.nodes.contains_key(node.id()) {
            return Err(SceneWorldError::ObjectAlreadyLive(node.id()));
        }
        for child in node.children() {
            self.validate_reserved_node(child)?;
        }
        Ok(())
    }

    fn materialize_reserved_node(
        &mut self,
        reserved: &ReservedNode,
        parent: Option<MobjectId>,
    ) -> Result<(), SceneWorldError> {
        let visual = match reserved.visual() {
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
        let node = MobjectNode::new(
            visual,
            reserved.name().to_owned(),
            parent,
            reserved.transform(),
            self.next_insertion_order,
        );
        self.next_insertion_order = self
            .next_insertion_order
            .checked_add(1)
            .expect("mobject insertion order overflowed");
        self.nodes.insert(reserved.id(), node);
        match parent {
            Some(parent) => self.get_mut(parent)?.add_child(reserved.id()),
            None => self.roots.push(reserved.id()),
        }
        for child in reserved.children() {
            self.materialize_reserved_node(child, Some(reserved.id()))?;
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
        Ok(self.ordered_ids(self.get(id)?.children()).into_owned())
    }

    pub fn rectangle(&self, id: MobjectId) -> Result<&Rectangle, SceneWorldError> {
        let rectangle = self.rectangle_component_id(id)?;
        Ok(&self.rectangles[rectangle].shape)
    }

    pub fn rectangle_geometry_revision(&self, id: MobjectId) -> Result<u64, SceneWorldError> {
        let rectangle = self.rectangle_component_id(id)?;
        Ok(self.rectangles[rectangle].geometry_revision)
    }

    pub fn set_rectangle_corners(
        &mut self,
        id: MobjectId,
        corners: [Point3<GMFloat>; 4],
    ) -> Result<(), SceneWorldError> {
        let rectangle = self.rectangle_component_id(id)?;
        let component = &mut self.rectangles[rectangle];
        component.shape.set_corners(corners);
        component.geometry_revision = component.geometry_revision.wrapping_add(1);
        component.dynamic = true;
        Ok(())
    }

    pub fn freeze_rectangle_geometry(&mut self, id: MobjectId) -> Result<(), SceneWorldError> {
        let rectangle = self.rectangle_component_id(id)?;
        self.rectangles[rectangle].dynamic = false;
        Ok(())
    }

    fn rectangle_component_id(&self, id: MobjectId) -> Result<RectangleId, SceneWorldError> {
        match self.get(id)?.visual() {
            VisualComponent::Rectangle(rectangle) => Ok(*rectangle),
            VisualComponent::None | VisualComponent::Renderable(_) => {
                Err(SceneWorldError::NotRectangle(id))
            }
        }
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
                ancestor = self.get(id)?.parent();
            }
        }

        let old_parent = self.get(child)?.parent();
        if old_parent == parent {
            return Ok(());
        }
        match old_parent {
            Some(old_parent) => self.get_mut(old_parent)?.remove_child(child),
            None => self.roots.retain(|id| *id != child),
        }
        self.get_mut(child)?.set_parent_link(parent);
        match parent {
            Some(parent) => self.get_mut(parent)?.add_child(child),
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
        let parent = self.get(id)?.parent();
        match parent {
            Some(parent) => self.get_mut(parent)?.remove_child(id),
            None => self.roots.retain(|root| *root != id),
        }

        let mut pending = vec![id];
        let mut removed = Vec::new();
        while let Some(current) = pending.pop() {
            let node = self
                .nodes
                .remove(current)
                .ok_or(SceneWorldError::ReservedObject(current))?;
            let (visual, children) = node.into_removal_parts();
            if let VisualComponent::Rectangle(rectangle) = visual {
                self.rectangles.remove(rectangle);
            }
            pending.extend(children);
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
            .find(|id| self.nodes[*id].name() == first)?;
        for component in components {
            current = self
                .children(current)
                .ok()?
                .into_iter()
                .find(|id| self.nodes[*id].name() == component)?;
        }
        Some(current)
    }

    pub fn submit_to_renderer(&self, visitor: &mut dyn RenderVisitor) {
        for root in self.ordered_ids(&self.roots).iter().copied() {
            self.submit_node(root, Matrix4::identity(), true, visitor);
        }
    }

    fn submit_node(
        &self,
        id: MobjectId,
        parent_transform: Matrix4<GMFloat>,
        parent_visible: bool,
        visitor: &mut dyn RenderVisitor,
    ) {
        let node = &self.nodes[id];
        let visible = parent_visible && node.visible();
        if !visible {
            return;
        }
        let world_transform = parent_transform * node.transform();
        match node.visual() {
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
        for child in self.ordered_ids(node.children()).iter().copied() {
            self.submit_node(child, world_transform, visible, visitor);
        }
    }

    fn ordered_ids<'a>(&self, ids: &'a [MobjectId]) -> Cow<'a, [MobjectId]> {
        let order = |id: MobjectId| self.nodes[id].order_key();
        if ids.windows(2).all(|pair| order(pair[0]) <= order(pair[1])) {
            return Cow::Borrowed(ids);
        }

        let mut sorted = ids.to_vec();
        sorted.sort_by_key(|id| order(*id));
        Cow::Owned(sorted)
    }
}

#[cfg(test)]
mod tests;
