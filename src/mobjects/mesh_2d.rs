use std::collections::hash_map::DefaultHasher;
use std::hash::Hasher;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::Color;
use lyon::tessellation::{
    FillVertex, FillVertexConstructor, StrokeVertex, StrokeVertexConstructor,
};

static NEXT_GEOMETRY_2D_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Geometry2DId(u64);

impl Geometry2DId {
    fn next() -> Self {
        Self(NEXT_GEOMETRY_2D_ID.fetch_add(1, Ordering::Relaxed))
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex2D {
    pub position: [f32; 2],
    /// Coordinates in the geometry's own local frame. For rectangles this is
    /// the rect frame (edges at +/- half extents); for generic lyon geometry
    /// it duplicates `position` and is unused by analytic AA.
    pub local: [f32; 2],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GeometryFingerprint {
    pub primary: u64,
    pub secondary: u64,
    pub vertex_count: u32,
    pub index_count: u32,
}

#[derive(Debug)]
pub struct MeshGeometry2D {
    id: Geometry2DId,
    vertices: Box<[Vertex2D]>,
    indices: Box<[u32]>,
    fingerprint: GeometryFingerprint,
}

impl MeshGeometry2D {
    pub fn new(vertices: Vec<Vertex2D>, indices: Vec<u32>) -> Self {
        let fingerprint = geometry_fingerprint(&vertices, &indices);
        Self {
            id: Geometry2DId::next(),
            vertices: vertices.into_boxed_slice(),
            indices: indices.into_boxed_slice(),
            fingerprint,
        }
    }

    pub fn id(&self) -> Geometry2DId {
        self.id
    }

    pub fn vertices(&self) -> &[Vertex2D] {
        &self.vertices
    }

    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    pub fn fingerprint(&self) -> GeometryFingerprint {
        self.fingerprint
    }

    pub fn same_geometry(&self, other: &Self) -> bool {
        if self.id == other.id {
            return true;
        }
        self.fingerprint == other.fingerprint
            && self
                .vertices
                .iter()
                .zip(other.vertices.iter())
                .all(|(left, right)| {
                    left.position[0].to_bits() == right.position[0].to_bits()
                        && left.position[1].to_bits() == right.position[1].to_bits()
                        && left.local[0].to_bits() == right.local[0].to_bits()
                        && left.local[1].to_bits() == right.local[1].to_bits()
                })
            && self.indices == other.indices
    }
}

#[derive(Clone, Debug)]
pub struct TriangleMesh2D {
    geometry: Arc<MeshGeometry2D>,
    color: [f32; 4],
    pub aa_mode: f32,
    analytic_half_extents: [f32; 2],
    dynamic: bool,
}

impl Default for TriangleMesh2D {
    fn default() -> Self {
        Self {
            geometry: Arc::new(MeshGeometry2D::new(Vec::new(), Vec::new())),
            color: color_to_f32(Color::default()),
            aa_mode: 0.0,
            analytic_half_extents: [0.0; 2],
            dynamic: false,
        }
    }
}

impl TriangleMesh2D {
    pub fn new(vertices: Vec<Vertex2D>, indices: Vec<u32>, color: Color) -> Self {
        Self {
            geometry: Arc::new(MeshGeometry2D::new(vertices, indices)),
            color: color_to_f32(color),
            aa_mode: 0.0,
            analytic_half_extents: [0.0; 2],
            dynamic: false,
        }
    }

    pub fn with_linear_color(mut self, color: [f32; 4]) -> Self {
        self.color = color;
        self
    }

    pub fn with_aa_params(mut self, aa_params: [f32; 4]) -> Self {
        self.analytic_half_extents = [aa_params[0], aa_params[1]];
        self.aa_mode = aa_params[2];
        self
    }

    pub fn with_dynamic(mut self, dynamic: bool) -> Self {
        self.dynamic = dynamic;
        self
    }

    pub fn set_dynamic(&mut self, dynamic: bool) {
        self.dynamic = dynamic;
    }

    pub fn is_dynamic(&self) -> bool {
        self.dynamic
    }

    pub fn aa_params(&self) -> [f32; 4] {
        [
            self.analytic_half_extents[0],
            self.analytic_half_extents[1],
            self.aa_mode,
            0.0,
        ]
    }

    pub fn replace_geometry(&mut self, vertices: Vec<Vertex2D>, indices: Vec<u32>, color: Color) {
        self.geometry = Arc::new(MeshGeometry2D::new(vertices, indices));
        self.color = color_to_f32(color);
    }

    pub fn geometry(&self) -> Arc<MeshGeometry2D> {
        self.geometry.clone()
    }

    pub fn vertices(&self) -> &[Vertex2D] {
        self.geometry.vertices()
    }

    pub fn indices(&self) -> &[u32] {
        self.geometry.indices()
    }

    pub fn color(&self) -> [f32; 4] {
        self.color
    }
}

pub struct VertexBuilder;

impl FillVertexConstructor<Vertex2D> for VertexBuilder {
    fn new_vertex(&mut self, vertex: FillVertex) -> Vertex2D {
        let position = [vertex.position().x, vertex.position().y];
        Vertex2D {
            position,
            local: position,
        }
    }
}

impl StrokeVertexConstructor<Vertex2D> for VertexBuilder {
    fn new_vertex(&mut self, vertex: StrokeVertex) -> Vertex2D {
        let position = [vertex.position().x, vertex.position().y];
        Vertex2D {
            position,
            local: position,
        }
    }
}

/// Projects tessellated vertices into a rectangle's own frame so the fragment
/// shader can compute the signed distance to the rect edges analytically.
///
/// `x_axis` and `y_axis` are the normalized rect edge directions; `local`
/// stores `dot(position - origin, axis)` for both axes.
#[derive(Clone, Copy)]
pub struct RectVertexBuilder {
    origin: [f32; 2],
    x_axis: [f32; 2],
    y_axis: [f32; 2],
}

impl RectVertexBuilder {
    pub fn new(origin: [f32; 2], x_axis: [f32; 2], y_axis: [f32; 2]) -> Self {
        Self {
            origin,
            x_axis,
            y_axis,
        }
    }

    pub fn project_vertex(&self, position: [f32; 2]) -> Vertex2D {
        Vertex2D {
            position,
            local: self.local_from_position(position),
        }
    }

    fn local_from_position(&self, position: [f32; 2]) -> [f32; 2] {
        let relative = [position[0] - self.origin[0], position[1] - self.origin[1]];
        [
            relative[0] * self.x_axis[0] + relative[1] * self.x_axis[1],
            relative[0] * self.y_axis[0] + relative[1] * self.y_axis[1],
        ]
    }
}

impl FillVertexConstructor<Vertex2D> for RectVertexBuilder {
    fn new_vertex(&mut self, vertex: FillVertex) -> Vertex2D {
        let position = [vertex.position().x, vertex.position().y];
        Vertex2D {
            position,
            local: self.local_from_position(position),
        }
    }
}

impl StrokeVertexConstructor<Vertex2D> for RectVertexBuilder {
    fn new_vertex(&mut self, vertex: StrokeVertex) -> Vertex2D {
        let position = [vertex.position().x, vertex.position().y];
        Vertex2D {
            position,
            local: self.local_from_position(position),
        }
    }
}

fn geometry_fingerprint(vertices: &[Vertex2D], indices: &[u32]) -> GeometryFingerprint {
    let vertex_bytes = bytemuck::cast_slice(vertices);
    let index_bytes = bytemuck::cast_slice(indices);

    let mut primary = DefaultHasher::new();
    primary.write(vertex_bytes);
    primary.write(index_bytes);

    let mut secondary = DefaultHasher::new();
    secondary.write_u64(0x9e37_79b9_7f4a_7c15);
    secondary.write(index_bytes);
    secondary.write(vertex_bytes);

    GeometryFingerprint {
        primary: primary.finish(),
        secondary: secondary.finish(),
        vertex_count: vertices.len() as u32,
        index_count: indices.len() as u32,
    }
}

fn color_to_f32(color: Color) -> [f32; 4] {
    [
        crate::srgb_to_linear(color.r),
        crate::srgb_to_linear(color.g),
        crate::srgb_to_linear(color.b),
        color.a as f32 / 255.0,
    ]
}

#[cfg(test)]
mod tests {
    use super::{MeshGeometry2D, Vertex2D};

    #[test]
    fn equal_geometry_has_equal_fingerprint() {
        let vertices = vec![
            Vertex2D {
                position: [0.0, 0.0],
                local: [0.0, 0.0],
            },
            Vertex2D {
                position: [1.0, 0.0],
                local: [1.0, 0.0],
            },
        ];
        let first = MeshGeometry2D::new(vertices.clone(), vec![0, 1]);
        let second = MeshGeometry2D::new(vertices, vec![0, 1]);

        assert_eq!(first.fingerprint(), second.fingerprint());
        assert!(first.same_geometry(&second));
    }

    #[test]
    fn geometry_equality_checks_content() {
        let first = MeshGeometry2D::new(
            vec![Vertex2D {
                position: [0.0, 0.0],
                local: [0.0, 0.0],
            }],
            vec![0],
        );
        let second = MeshGeometry2D::new(
            vec![Vertex2D {
                position: [-0.0, 0.0],
                local: [-0.0, 0.0],
            }],
            vec![0],
        );

        assert!(!first.same_geometry(&second));
    }
}
