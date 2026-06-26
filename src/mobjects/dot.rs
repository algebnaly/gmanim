use crate::mobjects::mesh_2d::{TriangleMesh2D, Vertex2D, VertexBuilder};
use lyon::math::point;
use lyon::path::Path;
use lyon::tessellation::{
    BuffersBuilder, FillOptions, FillTessellator, StrokeOptions, StrokeTessellator, VertexBuffers,
};
use nalgebra::Point3;
use std::f32::consts::PI;

use crate::{
    mobjects::{Draw, DrawConfig, Mobject, Transform},
    Color, Context, GMFloat,
};

pub struct Dot {
    pub position: Point3<GMFloat>,
    pub radius: GMFloat,
    pub color: Color,
    pub draw_config: DrawConfig,
    pub model_matrix: nalgebra::Matrix4<GMFloat>,
    pub mesh: TriangleMesh2D,
}

impl Default for Dot {
    fn default() -> Self {
        Self {
            position: Point3::origin(),
            radius: 0.05,
            color: Color::default(),
            draw_config: DrawConfig::default(),
            model_matrix: nalgebra::Matrix4::identity(),
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
            position,
            radius,
            color,
            draw_config,
            model_matrix: nalgebra::Matrix4::identity(),
            mesh: TriangleMesh2D::default(),
        }
    }
}

impl Draw for Dot {
    fn draw(&self, _ctx: &mut Context, _parent_matrix: nalgebra::Matrix4<GMFloat>) {}
}
impl Transform for Dot {
    fn get_model_matrix(&self) -> nalgebra::Matrix4<GMFloat> {
        self.model_matrix
    }
    fn set_model_matrix(&mut self, mat: nalgebra::Matrix4<GMFloat>) {
        self.model_matrix = mat;
    }
}

impl Mobject for Dot {
    fn as_mesh_2d(&self) -> Option<&TriangleMesh2D> {
        Some(&self.mesh)
    }
}
