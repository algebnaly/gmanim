use std::{fs, io::Read};

use nalgebra::Vector2;
use usvg::{Group, Node, tiny_skia_path::PathSegment};

use crate::{
    Context, GMFloat,
    math_utils::{point2d_to_point3d, point3d_to_point2d},
};

use super::{
    Draw, DrawConfig, Mobject, NodeBundle, coordinate_change_x, coordinate_change_y,
    path::PathElement,
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
    pub mesh: TriangleMesh2D,
}

impl SVGPath {
    pub fn new() -> Self {
        Self {
            elements: vec![],
            is_closed: false,
            draw_config: Default::default(),
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
        if self.draw_config.fill {
            let mut fill_tess = FillTessellator::new();
            let _ = fill_tess.tessellate_path(
                &path,
                &FillOptions::default(),
                &mut BuffersBuilder::new(&mut geometry, VertexBuilder),
            );
        }

        if self.draw_config.stoke_width > 0.0 {
            let mut stroke_tess = StrokeTessellator::new();
            let _ = stroke_tess.tessellate_path(
                &path,
                &StrokeOptions::default().with_line_width(self.draw_config.stoke_width as f32),
                &mut BuffersBuilder::new(&mut geometry, VertexBuilder),
            );
        }

        self.mesh
            .replace_geometry(geometry.vertices, geometry.indices, c);
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

impl Draw for SVGPath {
    fn draw(&self, _ctx: &mut crate::Context, _parent_matrix: nalgebra::Matrix4<crate::GMFloat>) {}
}

impl Mobject for SVGPath {
    fn default_name(&self) -> &'static str {
        "SVGPath"
    }

    fn submit_to_renderer(
        &self,
        visitor: &mut dyn crate::mobjects::RenderVisitor,
        world_transform: nalgebra::Matrix4<crate::GMFloat>,
    ) {
        visitor.push_mesh_2d(&self.mesh, world_transform);
    }
}

pub fn open_svg_file(svg_filepath: &str) -> NodeBundle {
    let mut svg_file = fs::File::options()
        .read(true)
        .open(svg_filepath)
        .expect("can't open svg file");
    let mut svg_str_buf = String::new();
    svg_file.read_to_string(&mut svg_str_buf);
    let tree = usvg::Tree::from_str(&svg_str_buf, &Default::default()).unwrap();
    let mut paths: Vec<SVGPath> = vec![];

    fn extract_paths(node: &Node, paths: &mut Vec<SVGPath>) {
        match node {
            Node::Group(g) => {
                for child in g.children() {
                    extract_paths(child, paths);
                }
            }
            Node::Path(path) => {
                //apply transform
                let mut svg_path = SVGPath::new();
                let transform = node.abs_transform();
                for e in path.data().segments() {
                    let pe = process_path_element(e, transform);
                    svg_path.elements.push(pe);
                }
                svg_path.flip_y_coordinate();
                svg_path.update_mesh();
                paths.push(svg_path);
            }
            _ => {}
        }
    }

    for node in tree.root().children() {
        extract_paths(&node, &mut paths);
    }

    let scaling_matrix =
        nalgebra::Matrix4::new_nonuniform_scaling(&nalgebra::Vector3::new(0.01, -0.01, 1.0));
    let mut root = NodeBundle::group("SvgRoot").with_transform(scaling_matrix);
    root.children = paths.into_iter().map(NodeBundle::new).collect();
    root
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
