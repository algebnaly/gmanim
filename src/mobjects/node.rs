use std::sync::Arc as Shared;

use nalgebra::{Matrix4, Vector3};
use slotmap::new_key_type;

use crate::GMFloat;

use super::mobject::Mobject;

new_key_type! {
    pub struct MobjectId;
    pub struct RectangleId;
}

#[derive(Clone)]
pub(super) enum VisualComponent {
    None,
    Renderable(Shared<dyn Mobject>),
    Rectangle(RectangleId),
}

#[derive(Clone)]
pub struct MobjectNode {
    name: String,
    parent: Option<MobjectId>,
    children: Vec<MobjectId>,
    transform: Matrix4<GMFloat>,
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
    pub(super) fn new(
        visual: VisualComponent,
        name: String,
        parent: Option<MobjectId>,
        transform: Matrix4<GMFloat>,
        insertion_order: u64,
    ) -> Self {
        Self {
            name,
            parent,
            children: Vec::new(),
            transform,
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

    pub fn transform(&self) -> Matrix4<GMFloat> {
        self.transform
    }

    pub fn set_transform(&mut self, transform: Matrix4<GMFloat>) {
        self.transform = transform;
    }

    pub fn apply_transform(&mut self, transform: Matrix4<GMFloat>) {
        self.transform = transform * self.transform;
    }

    pub fn move_by(&mut self, displacement: Vector3<GMFloat>) {
        self.apply_transform(Matrix4::new_translation(&displacement));
    }

    pub fn scale_by(&mut self, factor: GMFloat) {
        self.apply_transform(Matrix4::new_scaling(factor));
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

    pub(super) fn visual(&self) -> &VisualComponent {
        &self.visual
    }

    pub(super) fn add_child(&mut self, child: MobjectId) {
        self.children.push(child);
    }

    pub(super) fn remove_child(&mut self, child: MobjectId) {
        self.children.retain(|id| *id != child);
    }

    pub(super) fn set_parent_link(&mut self, parent: Option<MobjectId>) {
        self.parent = parent;
    }

    pub(super) fn order_key(&self) -> (i32, u64) {
        (self.layer, self.insertion_order)
    }

    pub(super) fn into_removal_parts(self) -> (VisualComponent, Vec<MobjectId>) {
        (self.visual, self.children)
    }
}
