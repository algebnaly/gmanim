pub mod basic;
pub mod dot;
pub mod formula;
pub mod grid_3d;
pub mod mesh_2d;
pub mod mesh_3d;
mod mobject;
mod node;
pub mod object_3d;
pub mod path;
pub mod polygon;
mod spawn;
mod submission;
pub mod svg_shape;
pub mod text;
mod world;
pub mod wrapper_3d;

pub use basic::{Arc, PolyLine, QuadraticBezier, Rectangle, SimpleLine};
pub use dot::Dot;
pub use grid_3d::{GridPlane, GridPlane3D, GridStyle3D};
pub use mesh_3d::{SphericalGridMaterial, SphericalPatchMaterial, SurfaceMaterial, TriangleMesh3D};
pub use mobject::Mobject;
pub use node::{MobjectId, MobjectNode, RectangleId};
pub use spawn::{NodeBundle, NodeVisual, SpawnPlan};
pub use submission::{Geometry3DRef, Grid3DSubmission, RenderVisitor, Surface3DSubmission};
pub use world::{SceneWorld, SceneWorldError};

use crate::{Color, GMFloat};

#[derive(Debug, Clone, Copy)]
pub struct DrawConfig {
    pub stoke_width: GMFloat,
    pub fill: bool,
    pub color: Color,
}

impl Default for DrawConfig {
    fn default() -> Self {
        DrawConfig {
            stoke_width: 0.25,
            fill: true,
            color: Default::default(),
        }
    }
}

#[inline]
pub fn coordinate_change_x(position_x: GMFloat, scene_width: GMFloat) -> GMFloat {
    scene_width / 2.0 + position_x
}

#[inline]
pub fn coordinate_change_y(position_y: GMFloat, scene_height: GMFloat) -> GMFloat {
    scene_height / 2.0 - position_y
}
