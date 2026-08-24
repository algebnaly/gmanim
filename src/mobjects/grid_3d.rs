use nalgebra::{Matrix4, Point3, Vector3};

use crate::{GMFloat, mobjects::Mobject};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GridPlane {
    Xy,
    Xz,
    Yz,
}

impl GridPlane {
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "xy" | "yx" => Some(Self::Xy),
            "xz" | "zx" => Some(Self::Xz),
            "yz" | "zy" => Some(Self::Yz),
            _ => None,
        }
    }

    pub fn axes(self) -> (Vector3<GMFloat>, Vector3<GMFloat>) {
        match self {
            Self::Xy => (Vector3::x(), Vector3::y()),
            Self::Xz => (Vector3::x(), Vector3::z()),
            Self::Yz => (Vector3::z(), Vector3::y()),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GridStyle3D {
    pub major_color: [f32; 4],
    pub minor_color: [f32; 4],
    pub u_axis_color: [f32; 4],
    pub v_axis_color: [f32; 4],
    pub cell_size: f32,
    pub subdivisions: u32,
    pub line_width_pixels: f32,
    pub fade_radius: f32,
}

impl Default for GridStyle3D {
    fn default() -> Self {
        Self {
            major_color: [0.35, 0.40, 0.48, 0.55],
            minor_color: [0.22, 0.26, 0.32, 0.30],
            u_axis_color: [0.85, 0.22, 0.22, 0.85],
            v_axis_color: [0.22, 0.52, 0.88, 0.85],
            cell_size: 1.0,
            subdivisions: 5,
            line_width_pixels: 1.2,
            fade_radius: 35.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GridPlane3D {
    pub plane: GridPlane,
    pub center: Point3<GMFloat>,
    pub size: GMFloat,
    pub style: GridStyle3D,
}

impl GridPlane3D {
    pub fn new(
        plane: GridPlane,
        center: Point3<GMFloat>,
        size: GMFloat,
        style: GridStyle3D,
    ) -> Self {
        assert!(size > 0.0, "grid size must be positive");
        assert!(style.cell_size > 0.0, "grid cell size must be positive");
        assert!(style.subdivisions > 0, "grid subdivisions must be non-zero");
        Self {
            plane,
            center,
            size,
            style,
        }
    }
}

impl Mobject for GridPlane3D {
    fn default_name(&self) -> &'static str {
        "GridPlane3D"
    }

    fn submit_to_renderer(
        &self,
        visitor: &mut dyn crate::mobjects::RenderVisitor,
        transform: Matrix4<GMFloat>,
    ) {
        visitor.push_grid_3d(crate::mobjects::Grid3DSubmission {
            grid: self,
            transform,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn named_planes_have_stable_axes() {
        assert_eq!(GridPlane::from_name("zx"), Some(GridPlane::Xz));
        assert_eq!(GridPlane::Xz.axes(), (Vector3::x(), Vector3::z()));
        assert_eq!(GridPlane::from_name("invalid"), None);
    }
}
