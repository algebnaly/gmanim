use crate::mobjects::get_2d_transform;
use nalgebra::Point3;
use crate::mobjects::mesh_2d::{TriangleMesh2D, Vertex2D, VertexBuilder};
use lyon::tessellation::{BuffersBuilder, FillOptions, FillTessellator, StrokeOptions, StrokeTessellator, VertexBuffers};
use lyon::path::Path;
use lyon::math::point;

use crate::{Color, Context, GMFloat, GMPoint, Scene};

use super::{Draw, DrawConfig, Mobject, Transform};

pub struct Polygon {
    pub vertices: Vec<GMPoint>,
    pub draw_config: DrawConfig,
    pub mesh: TriangleMesh2D,
    pub model_matrix: nalgebra::Matrix4<crate::GMFloat>,
}

impl Polygon {
    pub fn new(vertices: Vec<GMPoint>) -> Self {
        let mut p = Self {
            vertices,
            draw_config: DrawConfig::default(),
            mesh: TriangleMesh2D::default(),
            model_matrix: nalgebra::Matrix4::identity(),
        };
        p.update_mesh();
        p
    }
    
    pub fn update_mesh(&mut self) {
        let mut builder = Path::builder();
        if !self.vertices.is_empty() {
            let mut v_list = self.vertices.iter();
            let start = v_list.next().unwrap();
            builder.begin(point(start.x as f32, start.y as f32));
            for p in v_list {
                builder.line_to(point(p.x as f32, p.y as f32));
            }
            builder.end(true);
        } else {
            builder.begin(point(0.0, 0.0));
            builder.end(true);
        }
        let path = builder.build();

        let mut geometry: VertexBuffers<Vertex2D, u32> = VertexBuffers::new();
        let c = self.draw_config.color;
        let color = [c.r as f32 / 255.0, c.g as f32 / 255.0, c.b as f32 / 255.0, c.a as f32 / 255.0];
        
        if self.draw_config.fill {
            let mut fill_tess = FillTessellator::new();
            fill_tess.tessellate_path(
                &path,
                &FillOptions::default(),
                &mut BuffersBuilder::new(&mut geometry, VertexBuilder { color })
            ).unwrap();
        }

        if self.draw_config.stoke_width > 0.0 {
            let mut stroke_tess = StrokeTessellator::new();
            stroke_tess.tessellate_path(
                &path,
                &StrokeOptions::default().with_line_width(self.draw_config.stoke_width as f32),
                &mut BuffersBuilder::new(&mut geometry, VertexBuilder { color })
            ).unwrap();
        }

        self.mesh.vertices = geometry.vertices;
        self.mesh.indices = geometry.indices;
        self.mesh.model_matrix = self.model_matrix;
    }
}

impl Transform for Polygon {
    fn get_model_matrix(&self) -> nalgebra::Matrix4<crate::GMFloat> {
        self.model_matrix
    }
    fn set_model_matrix(&mut self, mat: nalgebra::Matrix4<crate::GMFloat>) {
        self.model_matrix = mat;
    }
}

impl Draw for Polygon {
    fn draw(&self, _ctx: &mut crate::Context, _parent_matrix: nalgebra::Matrix4<crate::GMFloat>) {
    }
}

impl Mobject for Polygon {
    fn as_mesh_2d(&self) -> Option<&TriangleMesh2D> {
        Some(&self.mesh)
    }
}

#[test]
pub fn test_polygon() {
    let mut ctx = Context::default();
    let mut scene = Scene::default();
    let v_list = vec![
        GMPoint::origin(),
        GMPoint::new(1.0, 1.0, 0.0),
        GMPoint::new(1.0, 2.0, 0.0),
    ];
    let mut polygon = Polygon::new(v_list);
    scene.add(Box::new(polygon));
    scene.save_png(&mut ctx, "output.png");
}
