use crate::Color;
use crate::mobjects::{Draw, Mobject, Transform};
use nalgebra::Matrix4;
use crate::GMFloat;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex2D {
    pub position: [f32; 2],
    pub color: [f32; 4],
}

pub struct TriangleMesh2D {
    pub vertices: Vec<Vertex2D>,
    pub indices: Vec<u32>,
    pub model_matrix: Matrix4<GMFloat>,
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
