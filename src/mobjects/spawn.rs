use std::sync::Arc as Shared;

use nalgebra::Matrix4;

use crate::GMFloat;

use super::{basic::Rectangle, mobject::Mobject, node::MobjectId};

#[derive(Clone)]
pub enum NodeVisual {
    None,
    Renderable(Shared<dyn Mobject>),
    Rectangle(Rectangle),
}

#[derive(Clone)]
pub struct NodeBundle {
    pub name: String,
    pub transform: Matrix4<GMFloat>,
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
            transform: Matrix4::identity(),
            visual: NodeVisual::Renderable(Shared::new(mobject)),
            children: Vec::new(),
        }
    }

    pub fn group(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            transform: Matrix4::identity(),
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
            transform: Matrix4::identity(),
            visual: NodeVisual::Rectangle(rectangle),
            children: Vec::new(),
        }
    }

    pub fn with_transform(mut self, transform: Matrix4<GMFloat>) -> Self {
        self.transform = transform;
        self
    }

    pub fn with_child(mut self, child: NodeBundle) -> Self {
        self.children.push(child);
        self
    }
}

#[derive(Clone)]
pub(super) struct ReservedNode {
    id: MobjectId,
    name: String,
    transform: Matrix4<GMFloat>,
    visual: NodeVisual,
    children: Vec<ReservedNode>,
}

impl ReservedNode {
    pub(super) fn from_bundle(
        bundle: NodeBundle,
        reserve_id: &mut impl FnMut() -> MobjectId,
    ) -> Self {
        let id = reserve_id();
        let children = bundle
            .children
            .into_iter()
            .map(|child| Self::from_bundle(child, reserve_id))
            .collect();
        Self {
            id,
            name: bundle.name,
            transform: bundle.transform,
            visual: bundle.visual,
            children,
        }
    }

    pub(super) fn id(&self) -> MobjectId {
        self.id
    }

    pub(super) fn name(&self) -> &str {
        &self.name
    }

    pub(super) fn transform(&self) -> Matrix4<GMFloat> {
        self.transform
    }

    pub(super) fn visual(&self) -> &NodeVisual {
        &self.visual
    }

    pub(super) fn children(&self) -> &[ReservedNode] {
        &self.children
    }
}

#[derive(Clone)]
pub struct SpawnPlan {
    world_id: u64,
    root: ReservedNode,
    parent: Option<MobjectId>,
}

impl SpawnPlan {
    pub(super) fn new(world_id: u64, root: ReservedNode, parent: Option<MobjectId>) -> Self {
        Self {
            world_id,
            root,
            parent,
        }
    }

    pub fn root(&self) -> MobjectId {
        self.root.id()
    }

    pub fn parent(&self) -> Option<MobjectId> {
        self.parent
    }

    pub fn ids(&self) -> Vec<MobjectId> {
        fn collect(node: &ReservedNode, ids: &mut Vec<MobjectId>) {
            ids.push(node.id());
            for child in node.children() {
                collect(child, ids);
            }
        }

        let mut ids = Vec::new();
        collect(&self.root, &mut ids);
        ids
    }

    pub(super) fn world_id(&self) -> u64 {
        self.world_id
    }

    pub(super) fn reserved_root(&self) -> &ReservedNode {
        &self.root
    }
}

impl std::fmt::Debug for SpawnPlan {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SpawnPlan")
            .field("world_id", &self.world_id)
            .field("root", &self.root.id())
            .field("parent", &self.parent)
            .field("ids", &self.ids())
            .finish()
    }
}
