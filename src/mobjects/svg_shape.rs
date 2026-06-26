use std::{fs, io::Read};

use nalgebra::Vector2;
use usvg::{tiny_skia_path::PathSegment, Group, Node};

use crate::{
    math_utils::{point2d_to_point3d, point3d_to_point2d},
    Context, GMFloat, Scene,
};

use super::{
    coordinate_change_x, coordinate_change_y, group::MobjectGroup, path::PathElement, Draw,
    DrawConfig, Mobject, Transform,
};
use crate::mobjects::mesh_2d::{TriangleMesh2D, Vertex2D, VertexBuilder};
use lyon::math::point;
use lyon::path::Path;
use lyon::tessellation::{
    BuffersBuilder, FillOptions, FillTessellator, StrokeOptions, StrokeTessellator, VertexBuffers,
};

#[derive(Debug)]
pub struct SVGPath {
    pub elements: Vec<PathElement>,
    pub is_closed: bool,
    pub draw_config: DrawConfig,
    pub model_matrix: nalgebra::Matrix4<crate::GMFloat>,
    pub mesh: TriangleMesh2D,
}

impl SVGPath {
    fn new() -> Self {
        Self {
            elements: vec![],
            is_closed: false,
            draw_config: Default::default(),
            model_matrix: nalgebra::Matrix4::identity(),
            mesh: TriangleMesh2D::default(),
        }
    }
    fn move_to_origin(&mut self) {
        if self.elements.len() == 0 {
            return;
        }
        let start = self.elements.first().unwrap();
        let start_pos;
        if let PathElement::MoveTo(p) = start {
            start_pos = p.clone();
        } else {
            return;
        }
        let start_displacement = nalgebra::Vector3::new(start_pos.x, start_pos.y, 0.0);
        for e in &mut self.elements {
            match e {
                PathElement::MoveTo(p) => {
                    *p -= start_displacement;
                }
                PathElement::LineTo(p) => {
                    *p -= start_displacement;
                }
                PathElement::QuadTo(p1, p2) => {
                    *p1 -= start_displacement;
                    *p2 -= start_displacement;
                }
                PathElement::CubicTo(p1, p2, p3) => {
                    *p1 -= start_displacement;
                    *p2 -= start_displacement;
                    *p3 -= start_displacement;
                }
                PathElement::Close => {}
            }
        }
    }
    pub fn update_mesh(&mut self) {
        let mut builder = Path::builder();
        let mut in_subpath = false;
        for e in &self.elements {
            match e {
                PathElement::MoveTo(p) => {
                    if in_subpath {
                        builder.end(false);
                    }
                    builder.begin(point(p.x as f32, p.y as f32));
                    in_subpath = true;
                }
                PathElement::LineTo(p) => {
                    builder.line_to(point(p.x as f32, p.y as f32));
                }
                PathElement::QuadTo(p1, p2) => {
                    builder.quadratic_bezier_to(
                        point(p1.x as f32, p1.y as f32),
                        point(p2.x as f32, p2.y as f32),
                    );
                }
                PathElement::CubicTo(p1, p2, p3) => {
                    builder.cubic_bezier_to(
                        point(p1.x as f32, p1.y as f32),
                        point(p2.x as f32, p2.y as f32),
                        point(p3.x as f32, p3.y as f32),
                    );
                }
                PathElement::Close => {
                    if in_subpath {
                        builder.end(true);
                        in_subpath = false;
                    }
                }
            }
        }
        if in_subpath {
            builder.end(self.is_closed);
        }
        let path = builder.build();

        let mut geometry: VertexBuffers<Vertex2D, u32> = VertexBuffers::new();
        let c = self.draw_config.color;
        let color = [
            c.r as f32 / 255.0,
            c.g as f32 / 255.0,
            c.b as f32 / 255.0,
            c.a as f32 / 255.0,
        ];

        if self.draw_config.fill {
            let mut fill_tess = FillTessellator::new();
            let _ = fill_tess.tessellate_path(
                &path,
                &FillOptions::default(),
                &mut BuffersBuilder::new(&mut geometry, VertexBuilder { color }),
            );
        }

        if self.draw_config.stoke_width > 0.0 {
            let mut stroke_tess = StrokeTessellator::new();
            let _ = stroke_tess.tessellate_path(
                &path,
                &StrokeOptions::default().with_line_width(self.draw_config.stoke_width as f32),
                &mut BuffersBuilder::new(&mut geometry, VertexBuilder { color }),
            );
        }

        self.mesh.vertices = geometry.vertices;
        self.mesh.indices = geometry.indices;
        self.mesh.model_matrix = self.model_matrix;
    }

    pub fn flip_y_coordinate(&mut self) {
        for e in &mut self.elements {
            match e {
                PathElement::MoveTo(p) => {
                    p.y = -p.y;
                }
                PathElement::LineTo(p) => {
                    p.y = -p.y;
                }
                PathElement::QuadTo(p1, p2) => {
                    p1.y = -p1.y;
                    p2.y = -p2.y;
                }
                PathElement::CubicTo(p1, p2, p3) => {
                    p1.y = -p1.y;
                    p2.y = -p2.y;
                    p3.y = -p3.y;
                }
                PathElement::Close => {}
            }
        }
    }
}

impl super::Transform for SVGPath {
    fn get_model_matrix(&self) -> nalgebra::Matrix4<crate::GMFloat> {
        self.model_matrix
    }
    fn set_model_matrix(&mut self, mat: nalgebra::Matrix4<crate::GMFloat>) {
        self.model_matrix = mat;
    }
}

impl Draw for SVGPath {
    fn draw(&self, _ctx: &mut crate::Context, _parent_matrix: nalgebra::Matrix4<crate::GMFloat>) {}
}

impl Mobject for SVGPath {
    fn as_mesh_2d(&self) -> Option<&TriangleMesh2D> {
        Some(&self.mesh)
    }
}

pub fn open_svg_file(svg_filepath: &str) -> MobjectGroup {
    let mut svg_file = fs::File::options()
        .read(true)
        .open(svg_filepath)
        .expect("can't open svg file");
    let mut svg_str_buf = String::new();
    svg_file.read_to_string(&mut svg_str_buf);
    let tree = usvg::Tree::from_str(&svg_str_buf, &Default::default()).unwrap();
    let mut paths: Vec<SVGPath> = vec![];
    for node in tree.root().children() {
        let n = node;
        match n {
            Node::Group(g) => {
                //we don't care for now
            }
            Node::Image(img) => {
                //we don't care for now
            }
            Node::Path(path) => {
                //apply transform
                let mut svg_path = SVGPath::new();
                let transform = node.abs_transform();
                let path_data = path;
                for e in path_data.data().segments() {
                    let pe = process_path_element(e, transform);
                    svg_path.elements.push(pe);
                }
                svg_path.flip_y_coordinate();
                svg_path.update_mesh();
                paths.push(svg_path);
            }
            Node::Text(_text) => {
                // we don't care for now
            }
            _ => {}
        }
    }

    let mut grp_mobj = MobjectGroup {
        mobjects: paths
            .into_iter()
            .map(|p| Box::new(p) as Box<dyn Mobject>)
            .collect(),
        model_matrix: nalgebra::Matrix4::identity(),
    };
    let scaling_matrix =
        nalgebra::Matrix4::new_nonuniform_scaling(&nalgebra::Vector3::new(0.01, -0.01, 1.0));
    grp_mobj.apply_transform(scaling_matrix);
    grp_mobj
}

fn map_point(
    transform: usvg::Transform,
    mut x: f32,
    mut y: f32,
) -> nalgebra::Point3<crate::GMFloat> {
    let tx = transform.sx * x + transform.kx * y + transform.tx;
    let ty = transform.ky * x + transform.sy * y + transform.ty;
    nalgebra::Point3::new(tx as crate::GMFloat, ty as crate::GMFloat, 0.0)
}

pub fn process_path_element(e: PathSegment, transform: usvg::Transform) -> PathElement {
    match e {
        PathSegment::MoveTo(p) => PathElement::MoveTo(map_point(transform, p.x as f32, p.y as f32)),
        PathSegment::LineTo(p) => PathElement::LineTo(map_point(transform, p.x as f32, p.y as f32)),
        PathSegment::QuadTo(p1, p2) => PathElement::QuadTo(
            map_point(transform, p1.x as f32, p1.y as f32),
            map_point(transform, p2.x as f32, p2.y as f32),
        ),
        PathSegment::CubicTo(p1, p2, p3) => PathElement::CubicTo(
            map_point(transform, p1.x as f32, p1.y as f32),
            map_point(transform, p2.x as f32, p2.y as f32),
            map_point(transform, p3.x as f32, p3.y as f32),
        ),
        PathSegment::Close => PathElement::Close,
    }
}
