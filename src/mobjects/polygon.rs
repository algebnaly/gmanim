use crate::GMPoint;
use crate::mobjects::mesh_2d::{TriangleMesh2D, Vertex2D, VertexBuilder};
use lyon::math::point;
use lyon::path::Path;
use lyon::tessellation::{
    BuffersBuilder, FillOptions, FillTessellator, StrokeOptions, StrokeTessellator, VertexBuffers,
};

use super::{DrawConfig, Mobject};

pub struct Polygon {
    pub vertices: Vec<GMPoint>,
    pub draw_config: DrawConfig,
    pub mesh: TriangleMesh2D,
}

impl Polygon {
    pub fn new(vertices: Vec<GMPoint>) -> Self {
        let mut p = Self {
            vertices,
            draw_config: DrawConfig::default(),
            mesh: TriangleMesh2D::default(),
        };
        p.update_mesh();
        p
    }

    pub fn update_mesh(&mut self) {
        let mut builder = Path::builder();
        if !self.vertices.is_empty() {
            let mut v_list = self.vertices.iter();
            let start = v_list.next().unwrap();
            builder.begin(point(start.x, start.y));
            for p in v_list {
                builder.line_to(point(p.x, p.y));
            }
            builder.end(true);
        } else {
            builder.begin(point(0.0, 0.0));
            builder.end(true);
        }
        let path = builder.build();

        let mut geometry: VertexBuffers<Vertex2D, u32> = VertexBuffers::new();
        let c = self.draw_config.color;
        if self.draw_config.fill {
            let mut fill_tess = FillTessellator::new();
            fill_tess
                .tessellate_path(
                    &path,
                    &FillOptions::default().with_tolerance(0.001),
                    &mut BuffersBuilder::new(&mut geometry, VertexBuilder),
                )
                .unwrap();
        }

        if self.draw_config.stoke_width > 0.0 {
            let mut stroke_tess = StrokeTessellator::new();
            stroke_tess
                .tessellate_path(
                    &path,
                    &StrokeOptions::default()
                        .with_line_width(self.draw_config.stoke_width)
                        .with_tolerance(0.001),
                    &mut BuffersBuilder::new(&mut geometry, VertexBuilder),
                )
                .unwrap();
        }

        self.mesh
            .replace_geometry(geometry.vertices, geometry.indices, c);
    }
}

impl Mobject for Polygon {
    fn default_name(&self) -> &'static str {
        "Polygon"
    }

    fn submit_to_renderer(
        &self,
        visitor: &mut dyn crate::mobjects::RenderVisitor,
        world_transform: nalgebra::Matrix4<crate::GMFloat>,
    ) {
        visitor.push_mesh_2d(&self.mesh, world_transform);
    }
}
