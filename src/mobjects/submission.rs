use nalgebra::Matrix4;

use crate::GMFloat;

use super::{
    RectangleId,
    basic::Rectangle,
    mesh_2d::TriangleMesh2D,
    mesh_3d::{SurfaceMaterial, TriangleMesh3D},
    object_3d::Object3D,
};

pub enum Geometry3DRef<'a> {
    Mesh(&'a TriangleMesh3D),
    Sdf(&'a dyn Object3D),
}

pub struct Surface3DSubmission<'a> {
    pub geometry: Geometry3DRef<'a>,
    pub material: SurfaceMaterial,
    pub transform: Matrix4<GMFloat>,
}

pub trait RenderVisitor {
    fn push_mesh_2d(&mut self, mesh: &TriangleMesh2D, transform: Matrix4<GMFloat>);

    fn push_rectangle_2d(
        &mut self,
        id: RectangleId,
        rectangle: &Rectangle,
        geometry_revision: u64,
        dynamic: bool,
        transform: Matrix4<GMFloat>,
    );

    fn push_surface_3d(&mut self, surface: Surface3DSubmission<'_>);
}
