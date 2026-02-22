use crate::GMFloat;

#[derive(Debug)]
pub enum PathElement {
    MoveTo(nalgebra::Point3<GMFloat>),
    LineTo(nalgebra::Point3<GMFloat>),
    QuadTo(nalgebra::Point3<GMFloat>, nalgebra::Point3<GMFloat>),
    CubicTo(
        nalgebra::Point3<GMFloat>,
        nalgebra::Point3<GMFloat>,
        nalgebra::Point3<GMFloat>,
    ),
    Close,
}

pub struct Path {}
