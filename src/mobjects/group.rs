use std::{cell::RefCell, rc::Rc};

use crate::{Context, GMFloat};

use super::{Draw, Mobject, Transform};

pub struct MobjectGroup {
    pub base: crate::mobjects::MobjectBase,
}

impl MobjectGroup {
    pub fn new() -> Self {
        Self {
            base: crate::mobjects::MobjectBase::new("MobjectGroup"),
        }
    }
}

impl Draw for MobjectGroup {
    fn draw(&self, _ctx: &mut crate::Context, _parent_matrix: nalgebra::Matrix4<crate::GMFloat>) {}
}

impl Mobject for MobjectGroup {
    fn submit_to_renderer(
        &self,
        visitor: &mut dyn crate::mobjects::RenderVisitor,
        parent_matrix: nalgebra::Matrix4<crate::GMFloat>,
    ) {
        let global_mat = parent_matrix * self.base.model_matrix;
        for child in self.base.children.iter() {
            child.borrow().submit_to_renderer(visitor, global_mat);
        }
    }
    fn base(&self) -> &crate::mobjects::MobjectBase {
        &self.base
    }
    fn base_mut(&mut self) -> &mut crate::mobjects::MobjectBase {
        &mut self.base
    }
}
