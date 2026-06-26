pub mod basic;
pub mod dot;
pub mod formula;
pub mod group;
pub mod mesh_2d;
pub mod mesh_3d;
pub mod object_3d;
pub mod path;
pub mod polygon;
pub mod scene_node;
pub mod svg_shape;
pub mod text;
pub mod three_d_viewport;
pub use basic::{Arc, PolyLine, Rectangle, SimpleLine};
pub use dot::Dot;

use crate::{Color, Context, GMFloat};

pub trait Mobject: Transform + Draw {
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
    fn as_3d(&self) -> Option<&dyn crate::mobjects::object_3d::Object3D> {
        None
    }
    fn as_mesh_2d(&self) -> Option<&crate::mobjects::mesh_2d::TriangleMesh2D> {
        None
    }
    fn as_mesh_3d(&self) -> Option<&crate::mobjects::mesh_3d::TriangleMesh3D> {
        None
    }
    fn as_scene_node(&self) -> Option<&crate::mobjects::scene_node::SceneNode> {
        None
    }
    fn get_name(&self) -> Option<String> {
        None
    }
    fn set_name(&mut self, _name: String) {}
    fn add_child(
        &mut self,
        _child: std::rc::Rc<std::cell::RefCell<Box<dyn crate::mobjects::Mobject>>>,
    ) {
    }
    fn remove_child(
        &mut self,
        _child: &std::rc::Rc<std::cell::RefCell<Box<dyn crate::mobjects::Mobject>>>,
    ) {
    }
}

pub trait MobjectClone: Mobject {
    fn mobject_clone(&self) -> Box<dyn MobjectClone>;
}

use nalgebra::Vector3;

pub trait Transform {
    // Modify the model_matrix natively
    fn get_model_matrix(&self) -> nalgebra::Matrix4<GMFloat>;
    fn set_model_matrix(&mut self, mat: nalgebra::Matrix4<GMFloat>);

    fn apply_transform(&mut self, transform: nalgebra::Matrix4<GMFloat>) {
        let current = self.get_model_matrix();
        self.set_model_matrix(transform * current);
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
