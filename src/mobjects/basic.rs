use nalgebra::Point3;
use std::f32::consts::PI;

use crate::{math_utils::k_for_bezier_arc, Color, Context, GMFloat, Scene};

use super::{Draw, DrawConfig, Mobject, Transform};
use crate::mobjects::mesh_2d::{TriangleMesh2D, Vertex2D, VertexBuilder};
use lyon::math::point;
use lyon::path::Path;
use lyon::tessellation::{
    BuffersBuilder, FillOptions, FillTessellator, StrokeOptions, StrokeTessellator, VertexBuffers,
};

pub struct Rectangle {
    pub p0: Point3<GMFloat>, // Top left
    pub p1: Point3<GMFloat>, // Bottom left
    pub p2: Point3<GMFloat>, // Bottom right
    pub p3: Point3<GMFloat>, // Top right
    pub color: Color,
    pub draw_config: DrawConfig,
    pub model_matrix: nalgebra::Matrix4<GMFloat>,
    pub mesh: TriangleMesh2D,
}

impl Default for Rectangle {
    fn default() -> Self {
        Rectangle {
            p0: Point3::new(0.0, 0.0, 0.0),
            p1: Point3::new(1.0, 0.0, 0.0),
            p2: Point3::new(1.0, 1.0, 0.0),
            p3: Point3::new(0.0, 1.0, 0.0),
            color: Color::default(),
            draw_config: DrawConfig::default(),
            model_matrix: nalgebra::Matrix4::identity(),
            mesh: TriangleMesh2D::default(),
        }
    }
}

impl Transform for Rectangle {
    fn get_model_matrix(&self) -> nalgebra::Matrix4<GMFloat> {
        self.model_matrix
    }
    fn set_model_matrix(&mut self, mat: nalgebra::Matrix4<GMFloat>) {
        self.model_matrix = mat;
        self.mesh.model_matrix = mat;
    }
}

impl Rectangle {
    pub fn update_mesh(&mut self) {
        let mut builder = Path::builder();
        builder.begin(point(self.p0.x as f32, self.p0.y as f32));
        builder.line_to(point(self.p1.x as f32, self.p1.y as f32));
        builder.line_to(point(self.p2.x as f32, self.p2.y as f32));
        builder.line_to(point(self.p3.x as f32, self.p3.y as f32));
        builder.end(true);
        let path = builder.build();

        let mut geometry: VertexBuffers<Vertex2D, u32> = VertexBuffers::new();
        let color = [
            self.color.r as f32 / 255.0,
            self.color.g as f32 / 255.0,
            self.color.b as f32 / 255.0,
            self.color.a as f32 / 255.0,
        ];

        if self.draw_config.fill {
            let mut fill_tess = FillTessellator::new();
            fill_tess
                .tessellate_path(
                    &path,
                    &FillOptions::default(),
                    &mut BuffersBuilder::new(&mut geometry, VertexBuilder { color }),
                )
                .unwrap();
        }

        if self.draw_config.stoke_width > 0.0 {
            let mut stroke_tess = StrokeTessellator::new();
            stroke_tess
                .tessellate_path(
                    &path,
                    &StrokeOptions::default().with_line_width(self.draw_config.stoke_width as f32),
                    &mut BuffersBuilder::new(&mut geometry, VertexBuilder { color }),
                )
                .unwrap();
        }

        self.mesh.vertices = geometry.vertices;
        self.mesh.indices = geometry.indices;
        self.mesh.model_matrix = self.model_matrix;
    }
}

impl Draw for Rectangle {
    fn draw(&self, _ctx: &mut Context, _parent_matrix: nalgebra::Matrix4<GMFloat>) {
        // empty
    }
}

impl Mobject for Rectangle {
    fn as_mesh_2d(&self) -> Option<&TriangleMesh2D> {
        Some(&self.mesh)
    }
}

#[derive(Default)]
pub struct SimpleLine {
    pub p0: Point3<GMFloat>,
    pub p1: Point3<GMFloat>,
    pub draw_config: DrawConfig,
    pub model_matrix: nalgebra::Matrix4<GMFloat>,
    pub mesh: TriangleMesh2D,
}

impl SimpleLine {
    pub fn new(p0: Point3<GMFloat>, p1: Point3<GMFloat>) -> Self {
        let mut sl = Self {
            p0,
            p1,
            draw_config: DrawConfig::default(),
            model_matrix: nalgebra::Matrix4::identity(),
            mesh: TriangleMesh2D::default(),
        };
        sl.update_mesh();
        sl
    }
}

impl Transform for SimpleLine {
    fn get_model_matrix(&self) -> nalgebra::Matrix4<GMFloat> {
        self.model_matrix
    }
    fn set_model_matrix(&mut self, mat: nalgebra::Matrix4<GMFloat>) {
        self.model_matrix = mat;
    }
}

impl SimpleLine {
    pub fn update_mesh(&mut self) {
        let mut builder = Path::builder();
        builder.begin(point(self.p0.x as f32, self.p0.y as f32));
        builder.line_to(point(self.p1.x as f32, self.p1.y as f32));
        builder.end(false);
        let path = builder.build();

        let mut geometry: VertexBuffers<Vertex2D, u32> = VertexBuffers::new();
        let c = self.draw_config.color;
        let color = [
            c.r as f32 / 255.0,
            c.g as f32 / 255.0,
            c.b as f32 / 255.0,
            c.a as f32 / 255.0,
        ];

        if self.draw_config.stoke_width > 0.0 {
            let mut stroke_tess = StrokeTessellator::new();
            stroke_tess
                .tessellate_path(
                    &path,
                    &StrokeOptions::default().with_line_width(self.draw_config.stoke_width as f32),
                    &mut BuffersBuilder::new(&mut geometry, VertexBuilder { color }),
                )
                .unwrap();
        }

        self.mesh.vertices = geometry.vertices;
        self.mesh.indices = geometry.indices;
        self.mesh.model_matrix = self.model_matrix;
    }
}

impl Draw for SimpleLine {
    fn draw(&self, _ctx: &mut Context, _parent_matrix: nalgebra::Matrix4<GMFloat>) {}
}

impl Mobject for SimpleLine {
    fn as_mesh_2d(&self) -> Option<&TriangleMesh2D> {
        Some(&self.mesh)
    }
}

#[derive(Default)]
pub struct PolyLine {
    pub points: Vec<Point3<GMFloat>>,
    pub draw_config: DrawConfig,
    pub model_matrix: nalgebra::Matrix4<GMFloat>,
    pub mesh: TriangleMesh2D,
}

impl PolyLine {
    pub fn new(points: Vec<Point3<GMFloat>>) -> Self {
        let mut pl = Self {
            points,
            draw_config: DrawConfig::default(),
            model_matrix: nalgebra::Matrix4::identity(),
            mesh: TriangleMesh2D::default(),
        };
        pl.update_mesh();
        pl
    }
}

impl Transform for PolyLine {
    fn get_model_matrix(&self) -> nalgebra::Matrix4<GMFloat> {
        self.model_matrix
    }
    fn set_model_matrix(&mut self, mat: nalgebra::Matrix4<GMFloat>) {
        self.model_matrix = mat;
    }
}

pub struct Arc {
    pub center_point: Point3<GMFloat>,
    pub start_angle: GMFloat,
    pub end_angle: GMFloat,
    pub radius: GMFloat,
    pub draw_config: DrawConfig,
    pub model_matrix: nalgebra::Matrix4<GMFloat>,
    pub mesh: TriangleMesh2D,
}

impl Arc {
    pub fn new(
        center_point: Point3<GMFloat>,
        start_angle: GMFloat,
        end_angle: GMFloat,
        radius: GMFloat,
    ) -> Self {
        let mut a = Self {
            center_point,
            start_angle,
            end_angle,
            radius,
            draw_config: DrawConfig::default(),
            model_matrix: nalgebra::Matrix4::identity(),
            mesh: TriangleMesh2D::default(),
        };
        a.update_mesh();
        a
    }
}

impl Arc {
    pub fn update_mesh(&mut self) {
        let mut builder = Path::builder();
        let num_curves =
            ((self.end_angle - self.start_angle).abs() / (PI as f32 / 2.0)).ceil() as usize;
        if num_curves == 0 {
            return;
        }

        let angle_step = (self.end_angle - self.start_angle) / (num_curves as f32);

        let start_x = self.center_point.x + self.radius * self.start_angle.cos();
        let start_y = self.center_point.y + self.radius * self.start_angle.sin();
        builder.begin(point(start_x as f32, start_y as f32));

        let mut current_angle = self.start_angle;
        for _ in 0..num_curves {
            let next_angle = current_angle + angle_step;
            let k = k_for_bezier_arc(angle_step);

            let cp1_x =
                self.center_point.x + self.radius * (current_angle.cos() - k * current_angle.sin());
            let cp1_y =
                self.center_point.y + self.radius * (current_angle.sin() + k * current_angle.cos());

            let cp2_x =
                self.center_point.x + self.radius * (next_angle.cos() + k * next_angle.sin());
            let cp2_y =
                self.center_point.y + self.radius * (next_angle.sin() - k * current_angle.cos());

            let end_x = self.center_point.x + self.radius * next_angle.cos();
            let end_y = self.center_point.y + self.radius * next_angle.sin();

            builder.cubic_bezier_to(
                point(cp1_x as f32, cp1_y as f32),
                point(cp2_x as f32, cp2_y as f32),
                point(end_x as f32, end_y as f32),
            );
            current_angle = next_angle;
        }
        builder.end(false);
        let path = builder.build();

        let mut geometry: VertexBuffers<Vertex2D, u32> = VertexBuffers::new();
        let c = self.draw_config.color;
        let color = [
            c.r as f32 / 255.0,
            c.g as f32 / 255.0,
            c.b as f32 / 255.0,
            c.a as f32 / 255.0,
        ];

        if self.draw_config.stoke_width > 0.0 {
            let mut stroke_tess = StrokeTessellator::new();
            stroke_tess
                .tessellate_path(
                    &path,
                    &StrokeOptions::default().with_line_width(self.draw_config.stoke_width as f32),
                    &mut BuffersBuilder::new(&mut geometry, VertexBuilder { color }),
                )
                .unwrap();
        }

        self.mesh.vertices = geometry.vertices;
        self.mesh.indices = geometry.indices;
        self.mesh.model_matrix = self.model_matrix;
    }
}

impl Draw for Arc {
    fn draw(&self, _ctx: &mut Context, _parent_matrix: nalgebra::Matrix4<GMFloat>) {}
}

impl Transform for Arc {
    fn get_model_matrix(&self) -> nalgebra::Matrix4<GMFloat> {
        self.model_matrix
    }
    fn set_model_matrix(&mut self, mat: nalgebra::Matrix4<GMFloat>) {
        self.model_matrix = mat;
    }
}

impl Mobject for Arc {
    fn as_mesh_2d(&self) -> Option<&TriangleMesh2D> {
        Some(&self.mesh)
    }
}

impl PolyLine {
    pub fn update_mesh(&mut self) {
        if self.points.is_empty() {
            return;
        }
        let mut builder = Path::builder();
        let mut first = true;
        for p in &self.points {
            if first {
                builder.begin(point(p.x as f32, p.y as f32));
                first = false;
            } else {
                builder.line_to(point(p.x as f32, p.y as f32));
            }
        }
        builder.end(false);
        let path = builder.build();

        let mut geometry: VertexBuffers<Vertex2D, u32> = VertexBuffers::new();
        let c = self.draw_config.color;
        let color = [
            c.r as f32 / 255.0,
            c.g as f32 / 255.0,
            c.b as f32 / 255.0,
            c.a as f32 / 255.0,
        ];

        if self.draw_config.stoke_width > 0.0 {
            let mut stroke_tess = StrokeTessellator::new();
            stroke_tess
                .tessellate_path(
                    &path,
                    &StrokeOptions::default().with_line_width(self.draw_config.stoke_width as f32),
                    &mut BuffersBuilder::new(&mut geometry, VertexBuilder { color }),
                )
                .unwrap();
        }

        self.mesh.vertices = geometry.vertices;
        self.mesh.indices = geometry.indices;
        self.mesh.model_matrix = self.model_matrix;
    }
}

impl Draw for PolyLine {
    fn draw(&self, _ctx: &mut Context, _parent_matrix: nalgebra::Matrix4<GMFloat>) {}
}

impl Mobject for PolyLine {
    fn as_mesh_2d(&self) -> Option<&TriangleMesh2D> {
        Some(&self.mesh)
    }
}

#[test]
fn test_draw_arc() {
    let mut ctx = Context::default();
    let mut scene = Scene::default();
    let arc = Arc::new(Point3::new(0.0, 1.0, 0.0), 0.0, PI as GMFloat * 2.0, 3.0);
    scene.add(Box::new(arc));
    scene.save_png(&mut ctx, "arc.png");
}
