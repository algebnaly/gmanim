use nalgebra::{Matrix4, Point3, Vector3};

use crate::{
    ClipRect, Color, EnvironmentLight, GMFloat, PointLight,
    camera::{CameraPose, OrthographicSetting, PerspectiveSetting, Projection},
};

use super::protocol::TrackValue;

impl TrackValue for Matrix4<GMFloat> {
    fn interpolate(from: &Self, to: &Self, alpha: GMFloat) -> Self {
        from * (1.0 - alpha) + to * alpha
    }
}

impl TrackValue for [Point3<GMFloat>; 4] {
    fn interpolate(from: &Self, to: &Self, alpha: GMFloat) -> Self {
        std::array::from_fn(|index| from[index] + (to[index] - from[index]) * alpha)
    }
}

impl TrackValue for bool {
    fn interpolate(from: &Self, to: &Self, alpha: GMFloat) -> Self {
        if alpha < 1.0 { *from } else { *to }
    }
}

impl TrackValue for i32 {
    fn interpolate(from: &Self, to: &Self, alpha: GMFloat) -> Self {
        ((*from as GMFloat) * (1.0 - alpha) + (*to as GMFloat) * alpha).round() as i32
    }
}

impl TrackValue for u32 {
    fn interpolate(from: &Self, to: &Self, alpha: GMFloat) -> Self {
        ((*from as GMFloat) * (1.0 - alpha) + (*to as GMFloat) * alpha).round() as u32
    }
}

impl TrackValue for CameraPose {
    fn interpolate(from: &Self, to: &Self, alpha: GMFloat) -> Self {
        Self {
            position: from.position + (to.position - from.position) * alpha,
            look_at: normalized_lerp(from.look_at, to.look_at, alpha),
            up_direction: normalized_lerp(from.up_direction, to.up_direction, alpha),
        }
    }
}

impl TrackValue for Projection {
    fn interpolate(from: &Self, to: &Self, alpha: GMFloat) -> Self {
        match (from, to) {
            (Self::Perspective(from), Self::Perspective(to)) => {
                let (fa, ff, fnr, ffr) = from.params();
                let (ta, tf, tnr, tfr) = to.params();
                Self::Perspective(PerspectiveSetting::new(
                    lerp(fa, ta, alpha),
                    lerp(ff, tf, alpha),
                    lerp(fnr, tnr, alpha),
                    lerp(ffr, tfr, alpha),
                ))
            }
            (Self::Orthographic(from), Self::Orthographic(to)) => {
                let (fl, fr, fb, ft, fnr, ffr) = from.params();
                let (tl, tr, tb, tt, tnr, tfr) = to.params();
                Self::Orthographic(OrthographicSetting::new(
                    lerp(fl, tl, alpha),
                    lerp(fr, tr, alpha),
                    lerp(fb, tb, alpha),
                    lerp(ft, tt, alpha),
                    lerp(fnr, tnr, alpha),
                    lerp(ffr, tfr, alpha),
                ))
            }
            _ if alpha < 1.0 => from.clone(),
            _ => to.clone(),
        }
    }
}

impl TrackValue for PointLight {
    fn interpolate(from: &Self, to: &Self, alpha: GMFloat) -> Self {
        Self {
            position: from.position + (to.position - from.position) * alpha,
            color: interpolate_color(from.color, to.color, alpha),
            intensity: lerp(from.intensity, to.intensity, alpha),
        }
    }
}

impl TrackValue for EnvironmentLight {
    fn interpolate(from: &Self, to: &Self, alpha: GMFloat) -> Self {
        Self {
            color: interpolate_color(from.color, to.color, alpha),
            intensity: lerp(from.intensity, to.intensity, alpha),
            rotation_radians: lerp(from.rotation_radians, to.rotation_radians, alpha),
        }
    }
}

impl TrackValue for Option<ClipRect> {
    fn interpolate(from: &Self, to: &Self, alpha: GMFloat) -> Self {
        match (from, to) {
            (Some(ClipRect::Logical(fx, fy, fw, fh)), Some(ClipRect::Logical(tx, ty, tw, th))) => {
                Some(ClipRect::Logical(
                    lerp_f32(*fx, *tx, alpha),
                    lerp_f32(*fy, *ty, alpha),
                    lerp_f32(*fw, *tw, alpha),
                    lerp_f32(*fh, *th, alpha),
                ))
            }
            (Some(ClipRect::Pixel(fx, fy, fw, fh)), Some(ClipRect::Pixel(tx, ty, tw, th))) => {
                Some(ClipRect::Pixel(
                    lerp_u32(*fx, *tx, alpha),
                    lerp_u32(*fy, *ty, alpha),
                    lerp_u32(*fw, *tw, alpha),
                    lerp_u32(*fh, *th, alpha),
                ))
            }
            _ if alpha < 1.0 => *from,
            _ => *to,
        }
    }
}

fn lerp(from: GMFloat, to: GMFloat, alpha: GMFloat) -> GMFloat {
    from * (1.0 - alpha) + to * alpha
}

fn lerp_f32(from: f32, to: f32, alpha: GMFloat) -> f32 {
    from * (1.0 - alpha) + to * alpha
}

fn lerp_u32(from: u32, to: u32, alpha: GMFloat) -> u32 {
    (from as f64 * (1.0 - alpha as f64) + to as f64 * alpha as f64).round() as u32
}

fn normalized_lerp(
    from: Vector3<GMFloat>,
    to: Vector3<GMFloat>,
    alpha: GMFloat,
) -> Vector3<GMFloat> {
    let value = from * (1.0 - alpha) + to * alpha;
    if value.norm_squared() > GMFloat::EPSILON {
        value.normalize()
    } else {
        to
    }
}

fn interpolate_color(from: Color, to: Color, alpha: GMFloat) -> Color {
    let channel = |from: u8, to: u8| {
        (from as GMFloat * (1.0 - alpha) + to as GMFloat * alpha)
            .round()
            .clamp(0.0, 255.0) as u8
    };
    Color::new(
        channel(from.r, to.r),
        channel(from.g, to.g),
        channel(from.b, to.b),
        channel(from.a, to.a),
    )
}
