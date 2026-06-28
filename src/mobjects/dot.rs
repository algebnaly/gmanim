use crate::mobjects::mesh_2d::{TriangleMesh2D, Vertex2D, VertexBuilder};
use lyon::math::point;
use lyon::path::Path;
use lyon::tessellation::{
    BuffersBuilder, FillOptions, FillTessellator, StrokeOptions, StrokeTessellator, VertexBuffers,
};
use nalgebra::Point3;
use std::f32::consts::PI;

use crate::{
    Color, Context, GMFloat,
    mobjects::{Draw, DrawConfig, Mobject, Transform},
};

pub struct Dot {
    pub base: crate::mobjects::MobjectBase,
    pub position: Point3<GMFloat>,
    pub radius: GMFloat,
    pub color: Color,
    pub draw_config: DrawConfig,
    pub mesh: TriangleMesh2D,
}

impl Default for Dot {
    fn default() -> Self {
        Self {
            base: crate::mobjects::MobjectBase::new("Dot"),
            position: Point3::origin(),
            radius: 0.05,
            color: Color::default(),
            draw_config: DrawConfig::default(),
            mesh: TriangleMesh2D::default(),
        }
    }
}

impl Dot {
    pub fn new(
        position: Point3<GMFloat>,
        radius: GMFloat,
        color: Color,
        draw_config: DrawConfig,
    ) -> Self {
        Self {
            base: crate::mobjects::MobjectBase::new("Dot"),
            position,
            radius,
            color,
            draw_config,
            mesh: TriangleMesh2D::default(),
        }
    }
}

impl Draw for Dot {
    fn draw(&self, _ctx: &mut Context, _parent_matrix: nalgebra::Matrix4<GMFloat>) {}
}

impl Mobject for Dot {
    fn submit_to_renderer(
        &self,
        visitor: &mut dyn crate::mobjects::RenderVisitor,
        parent_mat: nalgebra::Matrix4<crate::GMFloat>,
    ) {
        visitor.push_mesh_2d(&self.mesh, parent_mat * self.base.model_matrix);
        let global_mat = parent_mat * self.base.model_matrix;
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
