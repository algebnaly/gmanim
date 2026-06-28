pub mod basic;
pub mod dot;
pub mod formula;
pub mod group;
pub mod mesh_2d;
pub mod mesh_3d;
pub mod object_3d;
pub mod path;
pub mod polygon;
pub mod svg_shape;
pub mod text;
pub mod three_d_viewport;
pub use basic::{Arc, PolyLine, Rectangle, SimpleLine};
pub use dot::Dot;

use crate::{Color, Context, GMFloat};

use std::cell::RefCell;
use std::rc::Rc;

pub type MobjectRef = Rc<RefCell<dyn Mobject>>;

pub trait RenderVisitor {
    fn push_mesh_2d(
        &mut self,
        mesh: &crate::mobjects::mesh_2d::TriangleMesh2D,
        transform: nalgebra::Matrix4<crate::GMFloat>,
    );
    fn push_mesh_3d(
        &mut self,
        mesh: &crate::mobjects::mesh_3d::TriangleMesh3D,
        transform: nalgebra::Matrix4<crate::GMFloat>,
    );
    fn push_object_3d(
        &mut self,
        obj: &dyn crate::mobjects::object_3d::Object3D,
        transform: nalgebra::Matrix4<crate::GMFloat>,
    );
}

impl std::fmt::Debug for MobjectBase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MobjectBase")
            .field("name", &self.name)
            .finish()
    }
}

#[derive(Default)]
pub struct MobjectBase {
    pub name: String,
    pub children: Vec<MobjectRef>,
    pub model_matrix: nalgebra::Matrix4<GMFloat>,
}

impl MobjectBase {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            children: Vec::new(),
            model_matrix: nalgebra::Matrix4::identity(),
        }
    }
}

pub trait Mobject: Transform + Draw {
    fn base(&self) -> &MobjectBase;
    fn base_mut(&mut self) -> &mut MobjectBase;

    fn get_name(&self) -> &str {
        &self.base().name
    }

    fn set_name(&mut self, name: &str) {
        self.base_mut().name = name.to_string();
    }

    fn get_children(&self) -> &[MobjectRef] {
        &self.base().children
    }

    fn get_children_mut(&mut self) -> &mut [MobjectRef] {
        &mut self.base_mut().children
    }

    fn add_child(&mut self, child: MobjectRef) {
        self.base_mut().children.push(child);
    }

    fn remove_child(&mut self, child: &MobjectRef) {
        self.base_mut().children.retain(|c| !Rc::ptr_eq(c, child));
    }

    fn get_model_matrix(&self) -> nalgebra::Matrix4<GMFloat> {
        self.base().model_matrix
    }

    fn set_model_matrix(&mut self, mat: nalgebra::Matrix4<GMFloat>) {
        self.base_mut().model_matrix = mat;
    }

    fn get_position(&self) -> nalgebra::Point3<GMFloat> {
        let mat = self.get_model_matrix();
        nalgebra::Point3::new(mat.m14, mat.m24, mat.m34)
    }

    fn set_position(&mut self, pos: nalgebra::Point3<GMFloat>) {
        let mut mat = self.get_model_matrix();
        mat.m14 = pos.x;
        mat.m24 = pos.y;
        mat.m34 = pos.z;
        self.set_model_matrix(mat);
    }

    fn submit_to_renderer(
        &self,
        visitor: &mut dyn RenderVisitor,
        parent_mat: nalgebra::Matrix4<crate::GMFloat>,
    ) {
        let global_mat = parent_mat * self.get_model_matrix();
        for child in self.get_children() {
            child.borrow().submit_to_renderer(visitor, global_mat);
        }
    }

    fn get_by_path(&self, path: &str) -> Option<MobjectRef> {
        if path.is_empty() {
            return None;
        }
        let parts: Vec<&str> = path.splitn(2, '/').collect();
        let target_name = parts[0];

        for child in self.base().children.iter() {
            if child.borrow().get_name() == target_name {
                if parts.len() == 1 {
                    return Some(Rc::clone(child));
                } else {
                    return child.borrow().get_by_path(parts[1]);
                }
            }
        }
        None
    }
}

pub trait MobjectClone: Mobject {
    fn mobject_clone(&self) -> Box<dyn MobjectClone>;
}

use nalgebra::Vector3;

pub trait Transform {
    fn apply_transform(&mut self, transform: nalgebra::Matrix4<GMFloat>);
    fn move_this(&mut self, movement: nalgebra::Vector3<GMFloat>);
    fn scale(&mut self, scale_factor: GMFloat);
}

impl<T: Mobject + ?Sized> Transform for T {
    fn apply_transform(&mut self, transform: nalgebra::Matrix4<GMFloat>) {
        let current = self.base().model_matrix;
        self.base_mut().model_matrix = transform * current;
    }

    fn move_this(&mut self, movement: nalgebra::Vector3<GMFloat>) {
        let movement_matrix = nalgebra::Matrix4::new_translation(&movement);
        self.apply_transform(movement_matrix);
    }

    fn scale(&mut self, scale_factor: GMFloat) {
        let scaling_matrix = nalgebra::Matrix4::new_scaling(scale_factor);
        self.apply_transform(scaling_matrix);
    }
}

pub trait Draw {
    // draw shape, incorporating accumulated parent transformations
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
    //assume axis is a unit vector
}

#[inline]
pub fn coordinate_change_x(position_x: GMFloat, scene_width: GMFloat) -> GMFloat {
    scene_width / 2.0 + position_x
}

#[inline]
pub fn coordinate_change_y(position_y: GMFloat, scene_height: GMFloat) -> GMFloat {
    scene_height / 2.0 - position_y
}
pub mod wrapper_3d;
