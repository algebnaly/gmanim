pub mod basic;
pub mod dot;
pub mod formula;
pub mod group;
pub mod object_3d;
pub mod scene_node;
pub mod mesh_3d;
pub mod mesh_2d;
pub mod path;
pub mod polygon;
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
    fn as_mesh_3d(&self) -> Option<&crate::mobjects::mesh_3d::TriangleMesh3D> {
        None
    }
    fn as_mesh_2d(&self) -> Option<&crate::mobjects::mesh_2d::TriangleMesh2D> {
        None
    }
    fn as_scene_node(&self) -> Option<&crate::mobjects::scene_node::SceneNode> {
        None
    }
    fn get_name(&self) -> Option<String> { None }
    fn set_name(&mut self, _name: String) {}
    fn add_child(&mut self, _child: std::rc::Rc<std::cell::RefCell<Box<dyn crate::mobjects::Mobject>>>) {}
    fn remove_child(&mut self, _child: &std::rc::Rc<std::cell::RefCell<Box<dyn crate::mobjects::Mobject>>>) {}
}

pub trait MobjectClone: Mobject {
    fn mobject_clone(&self) -> Box<dyn MobjectClone>;
}



use nalgebra::Vector3;

pub fn get_2d_transform(ctx: &crate::Context, mat: nalgebra::Matrix4<crate::GMFloat>) -> tiny_skia::Transform {
    let math_transform = tiny_skia::Transform::from_row(
        mat.m11 as f32, mat.m21 as f32,
        mat.m12 as f32, mat.m22 as f32,
        mat.m14 as f32, mat.m24 as f32,
    );
    let math_to_screen = tiny_skia::Transform::from_row(
        ctx.scene_config.scale_factor, 0.0,
        0.0, -ctx.scene_config.scale_factor,
        (ctx.scene_config.width / 2.0) * ctx.scene_config.scale_factor,
        (ctx.scene_config.height / 2.0) * ctx.scene_config.scale_factor,
    );
    math_transform.post_concat(math_to_screen)
}



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

impl DrawConfig {
    pub fn get_stroke(&self, _scale_factor: f32) -> Option<tiny_skia::Stroke> {
        if self.stoke_width <= 0.0 {
            return None;
        }
        let mut stroke = tiny_skia::Stroke::default();
        stroke.width = self.stoke_width as f32; // Do not multiply by scale_factor here, tiny_skia applies ts_transform's scale automatically!
        stroke.line_cap = tiny_skia::LineCap::Round;
        stroke.line_join = tiny_skia::LineJoin::Round;
        Some(stroke)
    }
}

pub fn rotate_matrix(axis: Vector3<GMFloat>, theta: GMFloat) {
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
