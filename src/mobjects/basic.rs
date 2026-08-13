use nalgebra::Point3;
use std::f32::consts::PI;

use crate::{Color, Context, GMFloat, math_utils::k_for_bezier_arc};

use super::{Draw, DrawConfig, Mobject};
use crate::mobjects::mesh_2d::{RectVertexBuilder, TriangleMesh2D, Vertex2D, VertexBuilder};
use lyon::math::point;
use lyon::path::Path;
use lyon::tessellation::{
    BuffersBuilder, FillOptions, FillTessellator, StrokeOptions, StrokeTessellator, VertexBuffers,
};

#[derive(Clone, Debug)]
pub struct Rectangle {
    pub p0: Point3<GMFloat>, // Top left
    pub p1: Point3<GMFloat>, // Bottom left
    pub p2: Point3<GMFloat>, // Bottom right
    pub p3: Point3<GMFloat>, // Top right
    pub color: Color,
    pub draw_config: DrawConfig,
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
        }
    }
}

impl Rectangle {
    /// Tessellates the quad and stores per-vertex local coordinates in the
    /// rectangle's own frame (x along p0->p1, y along p0->p3, origin at the
    /// center). These coordinates feed the analytic edge-AA fragment shader;
    /// they are ignored by the legacy MSAA path.
    pub fn tessellate(&self) -> TriangleMesh2D {
        let center_x = (self.p0.x + self.p1.x + self.p2.x + self.p3.x) as f32 / 4.0;
        let center_y = (self.p0.y + self.p1.y + self.p2.y + self.p3.y) as f32 / 4.0;
        let edge_x = [
            (self.p1.x - self.p0.x) as f32,
            (self.p1.y - self.p0.y) as f32,
        ];
        let edge_y = [
            (self.p3.x - self.p0.x) as f32,
            (self.p3.y - self.p0.y) as f32,
        ];
        let edge_x_len = (edge_x[0] * edge_x[0] + edge_x[1] * edge_x[1]).sqrt().max(1e-8);
        let edge_y_len = (edge_y[0] * edge_y[0] + edge_y[1] * edge_y[1]).sqrt().max(1e-8);
        let rect_builder = RectVertexBuilder::new(
            [center_x, center_y],
            [edge_x[0] / edge_x_len, edge_x[1] / edge_x_len],
            [edge_y[0] / edge_y_len, edge_y[1] / edge_y_len],
        );

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
                    &FillOptions::default().with_tolerance(0.001),
                    &mut BuffersBuilder::new(&mut geometry, rect_builder),
                )
                .unwrap();
        }

        if self.draw_config.stoke_width > 0.0 {
            let mut stroke_tess = StrokeTessellator::new();
            stroke_tess
                .tessellate_path(
                    &path,
                    &StrokeOptions::default()
                        .with_line_width(self.draw_config.stoke_width as f32)
                        .with_tolerance(0.001),
                    &mut BuffersBuilder::new(&mut geometry, rect_builder),
                )
                .unwrap();
        }

        TriangleMesh2D::new(geometry.vertices, geometry.indices, self.color)
    }

    pub fn corners(&self) -> [Point3<GMFloat>; 4] {
        [self.p0, self.p1, self.p2, self.p3]
    }

    pub fn set_corners(&mut self, corners: [Point3<GMFloat>; 4]) {
        [self.p0, self.p1, self.p2, self.p3] = corners;
    }

    pub fn same_geometry(&self, other: &Self) -> bool {
        self.corners()
            .into_iter()
            .zip(other.corners())
            .all(|(left, right)| {
                left.x.to_bits() == right.x.to_bits()
                    && left.y.to_bits() == right.y.to_bits()
                    && left.z.to_bits() == right.z.to_bits()
            })
            && self.draw_config.fill == other.draw_config.fill
            && self.draw_config.stoke_width.to_bits() == other.draw_config.stoke_width.to_bits()
    }
}

#[derive(Default)]
pub struct SimpleLine {
    pub p0: Point3<GMFloat>,
    pub p1: Point3<GMFloat>,
    pub draw_config: DrawConfig,
    pub mesh: TriangleMesh2D,
    pub local_transform: nalgebra::Matrix4<GMFloat>,
}

impl SimpleLine {
    pub fn new(p0: Point3<GMFloat>, p1: Point3<GMFloat>) -> Self {
        let mut sl = Self {
            p0,
            p1,
            draw_config: DrawConfig::default(),
            mesh: TriangleMesh2D::default(),
            local_transform: nalgebra::Matrix4::identity(),
        };
        sl.update_mesh();
        sl
    }
}

impl SimpleLine {
    pub fn update_mesh(&mut self) {
        let mut u = self.p1 - self.p0;
        let u_len = u.norm();
        if u_len < 1e-6 {
            u = nalgebra::Vector3::new(1.0, 0.0, 0.0);
        } else {
            u /= u_len;
        }

        let arbitrary = if u.x.abs() > 0.5 {
            nalgebra::Vector3::new(0.0, 1.0, 0.0)
        } else {
            nalgebra::Vector3::new(1.0, 0.0, 0.0)
        };
        let n = u.cross(&arbitrary).normalize();
        let v = n.cross(&u).normalize();

        let mut local_transform = nalgebra::Matrix4::identity();
        local_transform.set_column(0, &nalgebra::Vector4::new(u.x, u.y, u.z, 0.0));
        local_transform.set_column(1, &nalgebra::Vector4::new(v.x, v.y, v.z, 0.0));
        local_transform.set_column(2, &nalgebra::Vector4::new(n.x, n.y, n.z, 0.0));
        local_transform.set_column(3, &nalgebra::Vector4::new(self.p0.x, self.p0.y, self.p0.z, 1.0));
        self.local_transform = local_transform;

        let p0_2d = nalgebra::Point2::new(0.0, 0.0);
        let p1_2d = nalgebra::Point2::new((self.p1 - self.p0).dot(&u), 0.0);

        let mut builder = Path::builder();
        builder.begin(point(p0_2d.x as f32, p0_2d.y as f32));
        builder.line_to(point(p1_2d.x as f32, p1_2d.y as f32));
        builder.end(false);
        let path = builder.build();

        let mut geometry: VertexBuffers<Vertex2D, u32> = VertexBuffers::new();
        let c = self.draw_config.color;
        if self.draw_config.stoke_width > 0.0 {
            let mut stroke_tess = StrokeTessellator::new();
            stroke_tess
                .tessellate_path(
                    &path,
                    &StrokeOptions::default()
                        .with_line_width(self.draw_config.stoke_width as f32)
                        .with_tolerance(0.001),
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
    fn default_name(&self) -> &'static str {
        "SimpleLine"
    }

    fn submit_to_renderer(
        &self,
        visitor: &mut dyn crate::mobjects::RenderVisitor,
        world_transform: nalgebra::Matrix4<crate::GMFloat>,
    ) {
        visitor.push_mesh_2d(&self.mesh, world_transform * self.local_transform);
    }
}

#[derive(Default)]
pub struct PolyLine {
    pub points: Vec<Point3<GMFloat>>,
    pub draw_config: DrawConfig,
    pub mesh: TriangleMesh2D,
}

impl PolyLine {
    pub fn new(points: Vec<Point3<GMFloat>>) -> Self {
        let mut pl = Self {
            points,
            draw_config: DrawConfig::default(),
            mesh: TriangleMesh2D::default(),
        };
        pl.update_mesh();
        pl
    }
}

pub struct Arc {
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
                    &StrokeOptions::default()
                        .with_line_width(self.draw_config.stoke_width as f32)
                        .with_tolerance(0.001),
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
    fn default_name(&self) -> &'static str {
        "Arc"
    }

    fn submit_to_renderer(
        &self,
        visitor: &mut dyn crate::mobjects::RenderVisitor,
        world_transform: nalgebra::Matrix4<crate::GMFloat>,
    ) {
        visitor.push_mesh_2d(&self.mesh, world_transform);
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
                        .with_line_width(self.draw_config.stoke_width as f32)
                        .with_tolerance(0.001),
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
    fn default_name(&self) -> &'static str {
        "PolyLine"
    }

    fn submit_to_renderer(
        &self,
        visitor: &mut dyn crate::mobjects::RenderVisitor,
        world_transform: nalgebra::Matrix4<crate::GMFloat>,
    ) {
        visitor.push_mesh_2d(&self.mesh, world_transform);
    }
}

pub struct QuadraticBezier {
    pub a: Point3<GMFloat>,
    pub b: Point3<GMFloat>,
    pub c: Point3<GMFloat>,
    pub draw_config: DrawConfig,
    pub mesh: TriangleMesh2D,
    pub local_transform: nalgebra::Matrix4<GMFloat>,
}

impl QuadraticBezier {
    pub fn new(a: Point3<GMFloat>, b: Point3<GMFloat>, c: Point3<GMFloat>) -> Self {
        let mut qb = Self {
            a,
            b,
            c,
            draw_config: DrawConfig::default(),
            mesh: TriangleMesh2D::default(),
            local_transform: nalgebra::Matrix4::identity(),
        };
        qb.update_mesh();
        qb
    }

    pub fn update_mesh(&mut self) {
        let mut u = self.c - self.a;
        let u_len = u.norm();
        if u_len < 1e-6 {
            u = nalgebra::Vector3::new(1.0, 0.0, 0.0);
        } else {
            u /= u_len;
        }

        let mut n = u.cross(&(self.b - self.a));
        if n.norm_squared() < 1e-6 {
            let arbitrary = if u.x.abs() > 0.5 {
                nalgebra::Vector3::new(0.0, 1.0, 0.0)
            } else {
                nalgebra::Vector3::new(1.0, 0.0, 0.0)
            };
            n = u.cross(&arbitrary).normalize();
        } else {
            n.normalize_mut();
        }
        let v = n.cross(&u).normalize();

        let mut local_transform = nalgebra::Matrix4::identity();
        local_transform.set_column(0, &nalgebra::Vector4::new(u.x, u.y, u.z, 0.0));
        local_transform.set_column(1, &nalgebra::Vector4::new(v.x, v.y, v.z, 0.0));
        local_transform.set_column(2, &nalgebra::Vector4::new(n.x, n.y, n.z, 0.0));
        local_transform.set_column(3, &nalgebra::Vector4::new(self.a.x, self.a.y, self.a.z, 1.0));
        self.local_transform = local_transform;

        let a_2d = nalgebra::Point2::new(0.0, 0.0);
        let b_2d = nalgebra::Point2::new((self.b - self.a).dot(&u), (self.b - self.a).dot(&v));
        let c_2d = nalgebra::Point2::new((self.c - self.a).dot(&u), (self.c - self.a).dot(&v));

        let mut builder = Path::builder();
        builder.begin(point(a_2d.x as f32, a_2d.y as f32));
        builder.quadratic_bezier_to(
            point(b_2d.x as f32, b_2d.y as f32),
            point(c_2d.x as f32, c_2d.y as f32),
        );
        builder.end(false);
        let path = builder.build();

        let mut geometry: VertexBuffers<Vertex2D, u32> = VertexBuffers::new();
        let color = self.draw_config.color;
        if self.draw_config.stoke_width > 0.0 {
            let mut stroke_tess = StrokeTessellator::new();
            stroke_tess
                .tessellate_path(
                    &path,
                    &StrokeOptions::default()
                        .with_line_width(self.draw_config.stoke_width as f32)
                        .with_tolerance(0.001),
                    &mut BuffersBuilder::new(&mut geometry, VertexBuilder),
                )
                .unwrap();
        }

        self.mesh
            .replace_geometry(geometry.vertices, geometry.indices, color);
    }
}

impl Draw for QuadraticBezier {
    fn draw(&self, _ctx: &mut Context, _parent_matrix: nalgebra::Matrix4<GMFloat>) {}
}

impl Mobject for QuadraticBezier {
    fn default_name(&self) -> &'static str {
        "QuadraticBezier"
    }

    fn submit_to_renderer(
        &self,
        visitor: &mut dyn crate::mobjects::RenderVisitor,
        world_transform: nalgebra::Matrix4<crate::GMFloat>,
    ) {
        visitor.push_mesh_2d(&self.mesh, world_transform * self.local_transform);
    }
}
