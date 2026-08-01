use std::sync::Arc;

use crate::mobjects::{
    Geometry3DRef, Mobject, Rectangle, RectangleId, RenderVisitor, Surface3DSubmission,
    mesh_2d::TriangleMesh2D, mesh_3d::TriangleMesh3D, mesh_3d::Vertex,
};
use nalgebra::Matrix4;

pub struct Wrapper2DIn3D {
    pub inner: Arc<dyn Mobject>,
}

impl Wrapper2DIn3D {
    pub fn new(inner: impl Mobject) -> Self {
        Self {
            inner: Arc::new(inner),
        }
    }
}

struct InterceptorVisitor<'a> {
    real_visitor: &'a mut dyn RenderVisitor,
    wrapper_matrix: Matrix4<crate::GMFloat>,
}

impl RenderVisitor for InterceptorVisitor<'_> {
    fn push_mesh_2d(&mut self, mesh: &TriangleMesh2D, transform: Matrix4<crate::GMFloat>) {
        let mut mesh3d = TriangleMesh3D::new(Vec::new(), Vec::new());
        let color = mesh.color();
        for vertex in mesh.vertices() {
            mesh3d.vertices.push(Vertex {
                position: [vertex.position[0], vertex.position[1], 0.0],
                normal: [0.0, 0.0, 1.0],
                color,
                surface_coord: [0.0, 0.0, 1.0],
            });
        }
        mesh3d.indices = mesh.indices().to_vec();
        mesh3d.material.unlit = true;

        self.real_visitor.push_surface_3d(Surface3DSubmission {
            geometry: Geometry3DRef::Mesh(&mesh3d),
            material: mesh3d.material,
            transform: self.wrapper_matrix * transform,
        });
    }

    fn push_rectangle_2d(
        &mut self,
        _id: RectangleId,
        rectangle: &Rectangle,
        _geometry_revision: u64,
        _dynamic: bool,
        transform: Matrix4<crate::GMFloat>,
    ) {
        self.push_mesh_2d(&rectangle.tessellate(), transform);
    }

    fn push_surface_3d(&mut self, surface: Surface3DSubmission<'_>) {
        self.real_visitor.push_surface_3d(Surface3DSubmission {
            geometry: surface.geometry,
            material: surface.material,
            transform: self.wrapper_matrix * surface.transform,
        });
    }
}

impl crate::mobjects::Draw for Wrapper2DIn3D {
    fn draw(&self, ctx: &mut crate::Context, world_transform: Matrix4<crate::GMFloat>) {
        self.inner.draw(ctx, world_transform);
    }
}

impl Mobject for Wrapper2DIn3D {
    fn default_name(&self) -> &'static str {
        "Wrapper2DIn3D"
    }

    fn submit_to_renderer(
        &self,
        visitor: &mut dyn RenderVisitor,
        world_transform: Matrix4<crate::GMFloat>,
    ) {
        let mut interceptor = InterceptorVisitor {
            real_visitor: visitor,
            wrapper_matrix: world_transform,
        };
        self.inner
            .submit_to_renderer(&mut interceptor, Matrix4::identity());
    }
}
