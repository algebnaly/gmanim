use crate::mobjects::{Draw, Mobject, Transform};
use crate::Color;
use crate::GMFloat;
use lyon::tessellation::{
    FillVertex, FillVertexConstructor, StrokeVertex, StrokeVertexConstructor,
};
use nalgebra::Matrix4;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex2D {
    pub position: [f32; 2],
    pub color: [f32; 4],
}

#[derive(Debug)]
pub struct TriangleMesh2D {
    pub vertices: Vec<Vertex2D>,
    pub indices: Vec<u32>,
    pub model_matrix: nalgebra::Matrix4<crate::GMFloat>,
}

impl Default for TriangleMesh2D {
    fn default() -> Self {
        Self {
            vertices: Vec::new(),
            indices: Vec::new(),
            model_matrix: nalgebra::Matrix4::identity(),
        }
    }
}

impl TriangleMesh2D {
    pub fn new(vertices: Vec<Vertex2D>, indices: Vec<u32>) -> Self {
        Self {
            vertices,
            indices,
            model_matrix: Matrix4::identity(),
        }
    }
}

pub struct VertexBuilder {
    pub color: [f32; 4],
}

impl FillVertexConstructor<Vertex2D> for VertexBuilder {
    fn new_vertex(&mut self, vertex: FillVertex) -> Vertex2D {
        Vertex2D {
            position: [vertex.position().x, vertex.position().y],
            color: self.color,
        }
    }
}

impl StrokeVertexConstructor<Vertex2D> for VertexBuilder {
    fn new_vertex(&mut self, vertex: StrokeVertex) -> Vertex2D {
        Vertex2D {
            position: [vertex.position().x, vertex.position().y],
            color: self.color,
        }
    }
}
