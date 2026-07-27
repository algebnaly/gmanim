use nalgebra::Point3;
use std::f32::consts::PI;

use crate::{Color, Context, GMFloat, Scene, math_utils::k_for_bezier_arc};

use super::{Draw, DrawConfig, Mobject, Transform};
use crate::mobjects::mesh_2d::{TriangleMesh2D, Vertex2D, VertexBuilder};
use lyon::math::point;
use lyon::path::Path;
use lyon::tessellation::{
    BuffersBuilder, FillOptions, FillTessellator, StrokeOptions, StrokeTessellator, VertexBuffers,
};

pub struct Rectangle {
    pub base: crate::mobjects::MobjectBase,
    pub p0: Point3<GMFloat>, // Top left
    pub p1: Point3<GMFloat>, // Bottom left
    pub p2: Point3<GMFloat>, // Bottom right
    pub p3: Point3<GMFloat>, // Top right
    pub color: Color,
    pub draw_config: DrawConfig,
    pub mesh: TriangleMesh2D,
}

impl Default for Rectangle {
    fn default() -> Self {
        Rectangle {
            base: crate::mobjects::MobjectBase::new("Rectangle"),
            p0: Point3::new(0.0, 0.0, 0.0),
            p1: Point3::new(1.0, 0.0, 0.0),
            p2: Point3::new(1.0, 1.0, 0.0),
            p3: Point3::new(0.0, 1.0, 0.0),
            color: Color::default(),
            draw_config: DrawConfig::default(),
            mesh: TriangleMesh2D::default(),
        }
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
        if self.draw_config.fill {
            let mut fill_tess = FillTessellator::new();
            fill_tess
                .tessellate_path(
                    &path,
                    &FillOptions::default(),
                    &mut BuffersBuilder::new(&mut geometry, VertexBuilder),
                )
                .unwrap();
        }

        if self.draw_config.stoke_width > 0.0 {
            let mut stroke_tess = StrokeTessellator::new();
            stroke_tess
                .tessellate_path(
                    &path,
                    &StrokeOptions::default().with_line_width(self.draw_config.stoke_width as f32),
                    &mut BuffersBuilder::new(&mut geometry, VertexBuilder),
                )
                .unwrap();
        }

        self.mesh
            .replace_geometry(geometry.vertices, geometry.indices, self.color);
    }
}

impl Draw for Rectangle {
    fn draw(&self, _ctx: &mut Context, _parent_matrix: nalgebra::Matrix4<GMFloat>) {
        // empty
    }
}

impl Mobject for Rectangle {
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

#[derive(Default)]
pub struct SimpleLine {
    pub base: crate::mobjects::MobjectBase,
    pub p0: Point3<GMFloat>,
    pub p1: Point3<GMFloat>,
    pub draw_config: DrawConfig,
    pub mesh: TriangleMesh2D,
}

impl SimpleLine {
    pub fn new(p0: Point3<GMFloat>, p1: Point3<GMFloat>) -> Self {
        let mut sl = Self {
            base: crate::mobjects::MobjectBase::new("SimpleLine"),
            p0,
            p1,
            draw_config: DrawConfig::default(),
            mesh: TriangleMesh2D::default(),
        };
        sl.update_mesh();
        sl
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
        if self.draw_config.stoke_width > 0.0 {
            let mut stroke_tess = StrokeTessellator::new();
            stroke_tess
                .tessellate_path(
                    &path,
                    &StrokeOptions::default().with_line_width(self.draw_config.stoke_width as f32),
                    &mut BuffersBuilder::new(&mut geometry, VertexBuilder),
                )
                .unwrap();
        }

        self.mesh
            .replace_geometry(geometry.vertices, geometry.indices, c);
    }
}

impl Draw for SimpleLine {
    fn draw(&self, _ctx: &mut Context, _parent_matrix: nalgebra::Matrix4<GMFloat>) {}
}

impl Mobject for SimpleLine {
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

#[derive(Default)]
pub struct PolyLine {
    pub base: crate::mobjects::MobjectBase,
    pub points: Vec<Point3<GMFloat>>,
    pub draw_config: DrawConfig,
    pub mesh: TriangleMesh2D,
}

impl PolyLine {
    pub fn new(points: Vec<Point3<GMFloat>>) -> Self {
        let mut pl = Self {
            base: crate::mobjects::MobjectBase::new("PolyLine"),
            points,
            draw_config: DrawConfig::default(),
            mesh: TriangleMesh2D::default(),
        };
        pl.update_mesh();
        pl
    }
}

pub struct Arc {
    pub base: crate::mobjects::MobjectBase,
    pub center_point: Point3<GMFloat>,
    pub start_angle: GMFloat,
    pub end_angle: GMFloat,
    pub radius: GMFloat,
    pub draw_config: DrawConfig,
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
            base: crate::mobjects::MobjectBase::new("Arc"),
            center_point,
            start_angle,
            end_angle,
            radius,
            draw_config: DrawConfig::default(),
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
        if self.draw_config.stoke_width > 0.0 {
            let mut stroke_tess = StrokeTessellator::new();
            stroke_tess
                .tessellate_path(
                    &path,
                    &StrokeOptions::default().with_line_width(self.draw_config.stoke_width as f32),
                    &mut BuffersBuilder::new(&mut geometry, VertexBuilder),
                )
                .unwrap();
        }

        self.mesh
            .replace_geometry(geometry.vertices, geometry.indices, c);
    }
}

impl Draw for Arc {
    fn draw(&self, _ctx: &mut Context, _parent_matrix: nalgebra::Matrix4<GMFloat>) {}
}

impl Mobject for Arc {
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
        builder.end(self.draw_config.fill);
        let path = builder.build();

        let mut geometry: VertexBuffers<Vertex2D, u32> = VertexBuffers::new();
        let c = self.draw_config.color;
        if self.draw_config.fill {
            let mut fill_tess = FillTessellator::new();
            fill_tess
                .tessellate_path(
                    &path,
                    &FillOptions::default(),
                    &mut BuffersBuilder::new(&mut geometry, VertexBuilder),
                )
                .unwrap();
        }

        if self.draw_config.stoke_width > 0.0 {
            let mut stroke_tess = StrokeTessellator::new();
            stroke_tess
                .tessellate_path(
                    &path,
                    &StrokeOptions::default().with_line_width(self.draw_config.stoke_width as f32),
                    &mut BuffersBuilder::new(&mut geometry, VertexBuilder),
                )
                .unwrap();
        }

        self.mesh
            .replace_geometry(geometry.vertices, geometry.indices, c);
    }
}

impl Draw for PolyLine {
    fn draw(&self, _ctx: &mut Context, _parent_matrix: nalgebra::Matrix4<GMFloat>) {}
}

impl Mobject for PolyLine {
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
