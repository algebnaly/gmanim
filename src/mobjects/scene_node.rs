use super::{Draw, Mobject, Transform};
use crate::{Context, GMFloat};
use nalgebra::Matrix4;
use std::cell::RefCell;
use std::rc::Rc;

pub struct SceneNode {
    pub name: String,
    pub local_transform: Matrix4<GMFloat>,
    pub children: Vec<Rc<RefCell<Box<dyn Mobject>>>>,
    pub component: Option<Box<dyn Mobject>>,
}

impl SceneNode {
    pub fn new(component: Option<Box<dyn Mobject>>) -> Self {
        Self {
            name: "Node".to_string(),
            local_transform: Matrix4::identity(),
            children: Vec::new(),
            component,
        }
    }

    pub fn empty() -> Self {
        Self::new(None)
    }

    pub fn add_child(&mut self, child: Rc<RefCell<Box<dyn Mobject>>>) {
        self.children.push(child);
    }

    pub fn remove_child(&mut self, child: &Rc<RefCell<Box<dyn Mobject>>>) {
        self.children.retain(|c| !Rc::ptr_eq(c, child));
    }
}

impl Transform for SceneNode {
    fn get_model_matrix(&self) -> Matrix4<GMFloat> {
        self.local_transform
    }

    fn set_model_matrix(&mut self, mat: Matrix4<GMFloat>) {
        self.local_transform = mat;
    }
}

impl Draw for SceneNode {
    fn draw(&self, ctx: &mut Context, parent_matrix: Matrix4<GMFloat>) {
        let global_mat = parent_matrix * self.local_transform;
        if let Some(comp) = &self.component {
            comp.draw(ctx, global_mat);
        }
        for child in &self.children {
            child.borrow().draw(ctx, global_mat);
        }
    }
}

impl Mobject for SceneNode {
    fn as_scene_node(&self) -> Option<&SceneNode> {
        Some(self)
    }
    fn get_name(&self) -> Option<String> {
        Some(self.name.clone())
    }
    fn set_name(&mut self, name: String) {
        self.name = name;
    }
    fn add_child(&mut self, child: Rc<RefCell<Box<dyn Mobject>>>) {
        self.children.push(child);
    }
    fn remove_child(&mut self, child: &Rc<RefCell<Box<dyn Mobject>>>) {
        self.children.retain(|c| !Rc::ptr_eq(c, child));
    }
}
