use crate::mobjects::{
    Geometry3DRef, Mobject, MobjectBase, RenderVisitor, Surface3DSubmission,
    mesh_2d::TriangleMesh2D, mesh_3d::TriangleMesh3D, mesh_3d::Vertex,
};
use nalgebra::Matrix4;

use std::cell::RefCell;
use std::rc::Rc;

pub struct Wrapper2DIn3D {
    pub base: MobjectBase,
    pub inner: Rc<RefCell<dyn Mobject>>,
}

impl Wrapper2DIn3D {
    pub fn new(name: &str, inner: Rc<RefCell<dyn Mobject>>) -> Self {
        Self {
            base: MobjectBase::new(name),
            inner,
        }
    }
}

struct InterceptorVisitor<'a> {
    real_visitor: &'a mut dyn RenderVisitor,
    wrapper_matrix: Matrix4<crate::GMFloat>,
}

impl<'a> RenderVisitor for InterceptorVisitor<'a> {
    fn push_mesh_2d(&mut self, mesh: &TriangleMesh2D, transform: Matrix4<crate::GMFloat>) {
        let mut mesh3d = TriangleMesh3D::new(Vec::new(), Vec::new());
        let color = mesh.color();
        for v in mesh.vertices() {
            mesh3d.vertices.push(Vertex {
                position: [v.position[0], v.position[1], 0.0],
                normal: [0.0, 0.0, 1.0],
                color,
                surface_coord: [0.0, 0.0, 1.0],
            });
        }
        mesh3d.indices = mesh.indices().to_vec();

        self.real_visitor.push_surface_3d(Surface3DSubmission {
            geometry: Geometry3DRef::Mesh(&mesh3d),
            material: mesh3d.material,
            transform: self.wrapper_matrix * transform,
        });
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
    fn draw(&self, ctx: &mut crate::mobjects::Context, parent_matrix: Matrix4<crate::GMFloat>) {
        self.inner
            .borrow()
            .draw(ctx, parent_matrix * self.base.model_matrix);
    }
}

impl Mobject for Wrapper2DIn3D {
    fn base(&self) -> &MobjectBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut MobjectBase {
        &mut self.base
    }

    fn submit_to_renderer(
        &self,
        visitor: &mut dyn RenderVisitor,
        parent_mat: Matrix4<crate::GMFloat>,
    ) {
        let mut interceptor = InterceptorVisitor {
            real_visitor: visitor,
            wrapper_matrix: parent_mat * self.base.model_matrix,
        };
        self.inner
            .borrow()
            .submit_to_renderer(&mut interceptor, Matrix4::identity());
    }
}
