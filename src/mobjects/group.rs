use std::{cell::RefCell, rc::Rc};

use crate::{Context, GMFloat};

use super::{Draw, Mobject, Transform};

pub struct MobjectGroup {
    pub mobjects: Vec<Box<dyn Mobject>>,
    pub model_matrix: nalgebra::Matrix4<GMFloat>,
}

impl MobjectGroup {
    pub fn new() -> Self {
        Self {
            mobjects: Vec::new(),
            model_matrix: nalgebra::Matrix4::identity(),
        }
    }
}

impl Transform for MobjectGroup {
    fn get_model_matrix(&self) -> nalgebra::Matrix4<GMFloat> {
        self.model_matrix
    }
    fn set_model_matrix(&mut self, mat: nalgebra::Matrix4<GMFloat>) {
        self.model_matrix = mat;
    }
}

impl Draw for MobjectGroup {
    fn draw(&self, ctx: &mut crate::Context, parent_matrix: nalgebra::Matrix4<GMFloat>) {
        let global_mat = parent_matrix * self.model_matrix;
        for m in &self.mobjects {
            m.draw(ctx, global_mat);
        }
    }
}

impl Mobject for MobjectGroup {}
