use nalgebra::{Point3, Vector3};
use rayon::prelude::*;
use std::sync::Arc;

use crate::camera::Camera;
use crate::mobjects::{Draw, Mobject, Transform};
use crate::{Color, Context, GMFloat};

// ═══════════════════════════════════════════════════════════════════════════
// 3D Object Trait
// ═══════════════════════════════════════════════════════════════════════════

/// Represents an object defined by a Signed Distance Function.
pub trait Object3D: Sync + Send {
    /// Returns the shortest distance from point `p` to the surface of the object (Used by CPU rendering).
    fn distance(&self, p: &Point3<GMFloat>) -> GMFloat;
    /// Returns the color of the object at point `p`.
    fn color(&self, p: &Point3<GMFloat>) -> Color;

    /// Returns the GPU-compatible primitive data
    fn as_primitive_data(
        &self,
        global_mat: nalgebra::Matrix4<GMFloat>,
    ) -> crate::vulkan::renderer::PrimitiveData3D;
}

// ═══════════════════════════════════════════════════════════════════════════
// Primitive 3D Objects
// ═══════════════════════════════════════════════════════════════════════════

/// A 3D Sphere
pub struct Sphere3D {
    pub center: Point3<GMFloat>,
    pub radius: GMFloat,
    pub color: Color,
    pub model_matrix: nalgebra::Matrix4<GMFloat>,
}

impl Object3D for Sphere3D {
    fn distance(&self, p: &Point3<GMFloat>) -> GMFloat {
        (p - self.center).norm() - self.radius
    }
    fn color(&self, _p: &Point3<GMFloat>) -> Color {
        self.color
    }
    fn as_primitive_data(
        &self,
        global_mat: nalgebra::Matrix4<GMFloat>,
    ) -> crate::vulkan::renderer::PrimitiveData3D {
        let center = global_mat.transform_point(&self.center);
        crate::vulkan::renderer::PrimitiveData3D {
            color: [
                self.color.r as f32 / 255.0,
                self.color.g as f32 / 255.0,
                self.color.b as f32 / 255.0,
                self.color.a as f32 / 255.0,
            ],
            params: [
                center.x as f32,
                center.y as f32,
                center.z as f32,
                self.radius as f32,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            shape_type: 0,
            padding: [0; 3],
        }
    }
}

impl Transform for Sphere3D {
    fn get_model_matrix(&self) -> nalgebra::Matrix4<GMFloat> {
        self.model_matrix
    }
    fn set_model_matrix(&mut self, mat: nalgebra::Matrix4<GMFloat>) {
        self.model_matrix = mat;
    }
}
impl Draw for Sphere3D {
    fn draw(&self, _ctx: &mut Context, _parent_matrix: nalgebra::Matrix4<GMFloat>) {}
}
impl Mobject for Sphere3D {
    fn as_3d(&self) -> Option<&dyn crate::mobjects::object_3d::Object3D> {
        Some(self)
    }

    fn set_position(&mut self, pos: nalgebra::Point3<GMFloat>) {
        self.center = pos;
    }

    fn get_position(&self) -> nalgebra::Point3<GMFloat> {
        self.center
    }
}

/// A 3D Line Segment (Capsule)
pub struct LineSegment3D {
    pub a: Point3<GMFloat>,
    pub b: Point3<GMFloat>,
    pub radius: GMFloat,
    pub color: Color,
    pub model_matrix: nalgebra::Matrix4<GMFloat>,
}

impl Object3D for LineSegment3D {
    fn distance(&self, p: &Point3<GMFloat>) -> GMFloat {
        let pa = p - self.a;
        let ba = self.b - self.a;
        let h = (pa.dot(&ba) / ba.dot(&ba)).clamp(0.0, 1.0);
        (pa - ba * h).norm() - self.radius
    }
    fn color(&self, _p: &Point3<GMFloat>) -> Color {
        self.color
    }
    fn as_primitive_data(
        &self,
        global_mat: nalgebra::Matrix4<GMFloat>,
    ) -> crate::vulkan::renderer::PrimitiveData3D {
        let a = global_mat.transform_point(&self.a);
        let b = global_mat.transform_point(&self.b);
        crate::vulkan::renderer::PrimitiveData3D {
            color: [
                self.color.r as f32 / 255.0,
                self.color.g as f32 / 255.0,
                self.color.b as f32 / 255.0,
                self.color.a as f32 / 255.0,
            ],
            params: [
                a.x as f32,
                a.y as f32,
                a.z as f32,
                b.x as f32,
                b.y as f32,
                b.z as f32,
                self.radius as f32,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            shape_type: 1,
            padding: [0; 3],
        }
    }
}

impl Transform for LineSegment3D {
    fn get_model_matrix(&self) -> nalgebra::Matrix4<GMFloat> {
        self.model_matrix
    }
    fn set_model_matrix(&mut self, mat: nalgebra::Matrix4<GMFloat>) {
        self.model_matrix = mat;
    }
}
impl Draw for LineSegment3D {
    fn draw(&self, _ctx: &mut Context, _parent_matrix: nalgebra::Matrix4<GMFloat>) {}
}
impl Mobject for LineSegment3D {
    fn as_3d(&self) -> Option<&dyn crate::mobjects::object_3d::Object3D> {
        Some(self)
    }
}

/// Inigo Quilez capped cone SDF in local coords: axial from 0 (base) to h (tip), radial distance.
fn sd_capped_cone_local(axial: GMFloat, radial: GMFloat, h: GMFloat, r: GMFloat) -> GMFloat {
    let q = nalgebra::Vector2::new(radial, axial);
    let k1 = nalgebra::Vector2::new(r, h);
    let k2 = nalgebra::Vector2::new(r - h, h);
    let cap_r = if q.y < 0.0 { r } else { q.y * r / h };
    let ca = nalgebra::Vector2::new(q.x - q.x.min(cap_r), q.y.abs() - h);
    let t = ((k2 - q).dot(&k1) / k1.dot(&k1)).clamp(0.0, 1.0);
    let cb = q - k2 + k1 * t;
    let s = if cb.x < 0.0 && ca.y < 0.0 { -1.0 } else { 1.0 };
    s * ca.dot(&ca).min(cb.dot(&cb)).sqrt()
}

/// A 3D Arrow (Union of a line segment and a cone)
pub struct Arrow3D {
    pub start: Point3<GMFloat>,
    pub end: Point3<GMFloat>,
    pub shaft_radius: GMFloat,
    pub head_radius: GMFloat,
    pub head_length: GMFloat,
    pub color: Color,
    pub model_matrix: nalgebra::Matrix4<GMFloat>,
}

impl Object3D for Arrow3D {
    fn distance(&self, p: &Point3<GMFloat>) -> GMFloat {
        let ba = self.end - self.start;
        let len = ba.norm();
        if len < 0.0001 {
            return (p - self.start).norm() - self.shaft_radius;
        }
        let dir = ba / len;
        let head_base = self.end - dir * self.head_length;

        // 1. Shaft (Capsule from start to head_base)
        let pa_s = p - self.start;
        let ba_s = head_base - self.start;
        let h_s = (pa_s.dot(&ba_s) / ba_s.dot(&ba_s)).clamp(0.0, 1.0);
        let d_shaft = (pa_s - ba_s * h_s).norm() - self.shaft_radius;

        // 2. Head (Capped cone from head_base to end)
        let q = p - head_base;
        let x = q.dot(&dir);
        let cr = (q - dir * x).norm();
        let d_cone = sd_capped_cone_local(x, cr, self.head_length, self.head_radius);

        // 3. Union (Min)
        d_shaft.min(d_cone)
    }

    fn color(&self, _p: &Point3<GMFloat>) -> Color {
        self.color
    }

    fn as_primitive_data(
        &self,
        global_mat: nalgebra::Matrix4<GMFloat>,
    ) -> crate::vulkan::renderer::PrimitiveData3D {
        let start = global_mat.transform_point(&self.start);
        let end = global_mat.transform_point(&self.end);
        crate::vulkan::renderer::PrimitiveData3D {
            color: [
                self.color.r as f32 / 255.0,
                self.color.g as f32 / 255.0,
                self.color.b as f32 / 255.0,
                self.color.a as f32 / 255.0,
            ],
            params: [
                start.x as f32,
                start.y as f32,
                start.z as f32,
                end.x as f32,
                end.y as f32,
                end.z as f32,
                self.shaft_radius as f32,
                self.head_radius as f32,
                self.head_length as f32,
                0.0,
                0.0,
                0.0,
            ],
            shape_type: 2,
            padding: [0; 3],
        }
    }
}

impl Transform for Arrow3D {
    fn get_model_matrix(&self) -> nalgebra::Matrix4<GMFloat> {
        self.model_matrix
    }
    fn set_model_matrix(&mut self, mat: nalgebra::Matrix4<GMFloat>) {
        self.model_matrix = mat;
    }
}
impl Draw for Arrow3D {
    fn draw(&self, _ctx: &mut Context, _parent_matrix: nalgebra::Matrix4<GMFloat>) {}
}
impl Mobject for Arrow3D {
    fn as_3d(&self) -> Option<&dyn crate::mobjects::object_3d::Object3D> {
        Some(self)
    }
}

pub struct Box3DSdf {
    pub center: Point3<GMFloat>,
    pub size: Vector3<GMFloat>,
    pub x_axis: Vector3<GMFloat>,
    pub y_axis: Vector3<GMFloat>,
    pub z_axis: Vector3<GMFloat>,
    pub color: Color,
    pub model_matrix: nalgebra::Matrix4<GMFloat>,
}

impl Object3D for Box3DSdf {
    fn distance(&self, p: &Point3<GMFloat>) -> GMFloat {
        let pt = p - self.center;
        let local_p = Vector3::new(
            pt.dot(&self.x_axis),
            pt.dot(&self.y_axis),
            pt.dot(&self.z_axis),
        );
        let d = Vector3::new(
            local_p.x.abs() - self.size.x,
            local_p.y.abs() - self.size.y,
            local_p.z.abs() - self.size.z,
        );

        let d_max = Vector3::new(d.x.max(0.0), d.y.max(0.0), d.z.max(0.0));
        let max_comp = d.x.max(d.y).max(d.z).min(0.0);
        d_max.norm() + max_comp
    }

    fn color(&self, _p: &Point3<GMFloat>) -> Color {
        self.color
    }

    fn as_primitive_data(
        &self,
        global_mat: nalgebra::Matrix4<GMFloat>,
    ) -> crate::vulkan::renderer::PrimitiveData3D {
        let center = global_mat.transform_point(&self.center);
        let x_axis = global_mat.transform_vector(&self.x_axis);
        let y_axis = global_mat.transform_vector(&self.y_axis);
        let z_axis = global_mat.transform_vector(&self.z_axis);
        crate::vulkan::renderer::PrimitiveData3D {
            color: [
                self.color.r as f32 / 255.0,
                self.color.g as f32 / 255.0,
                self.color.b as f32 / 255.0,
                self.color.a as f32 / 255.0,
            ],
            params: [
                center.x as f32,
                center.y as f32,
                center.z as f32,
                self.size.x as f32,
                self.size.y as f32,
                self.size.z as f32,
                x_axis.x as f32,
                x_axis.y as f32,
                x_axis.z as f32,
                y_axis.x as f32,
                y_axis.y as f32,
                y_axis.z as f32,
            ],
            shape_type: 3,
            padding: [0; 3],
        }
    }
}

impl Transform for Box3DSdf {
    fn get_model_matrix(&self) -> nalgebra::Matrix4<GMFloat> {
        self.model_matrix
    }
    fn set_model_matrix(&mut self, mat: nalgebra::Matrix4<GMFloat>) {
        self.model_matrix = mat;
    }
}

impl Draw for Box3DSdf {
    fn draw(&self, _ctx: &mut Context, _parent_matrix: nalgebra::Matrix4<GMFloat>) {}
}

impl Mobject for Box3DSdf {
    fn as_3d(&self) -> Option<&dyn crate::mobjects::object_3d::Object3D> {
        Some(self)
    }

    fn set_position(&mut self, pos: nalgebra::Point3<GMFloat>) {
        self.center = pos;
    }

    fn get_position(&self) -> nalgebra::Point3<GMFloat> {
        self.center
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::animation::{Rotate, Timeline, Wait};
    use crate::camera::{Camera, PerspectiveSetting, Projection};
    use crate::video_backend::ffmpeg::FfmpegBackend;
    use crate::{
        mobjects::SimpleLine,
        video_backend::{vaapi::FfmpegVaapiBackend, ColorOrder, VideoConfig},
        Context, Scene, SceneConfig,
    };

    #[test]
    fn test_scene_3d() {
        let width = 1920u32;
        let height = 1080u32;

        let ctx = Context {
            pixmap: Pixmap::new(width, height).unwrap(),
            scene_config: SceneConfig {
                width: 16.0,
                height: 9.0,
                output_width: width,
                output_height: height,
                scale_factor: height as GMFloat / 16.0,
            },
        };

        let mut scene = Scene::default();
        scene.camera.position = Point3::new(3.0, 3.0, 5.0);
        scene.camera.set_look_at(Vector3::new(-3.0, -3.0, -5.0)); // look at origin

        let arrow_ref = scene.add(Box::new(Arrow3D {
            start: Point3::new(-1.0, 0.0, 0.0),
            end: Point3::new(2.0, 1.0, 0.0),
            shaft_radius: 0.1,
            head_radius: 0.3,
            head_length: 0.6,
            color: Color::new(255, 50, 50, 255), // Red
            model_matrix: nalgebra::Matrix4::identity(),
        }));

        // Add 3D Line segment
        scene.add(Box::new(LineSegment3D {
            a: Point3::new(-2.0, -1.0, -1.0),
            b: Point3::new(1.0, -2.0, 1.0),
            radius: 0.15,
            color: Color::new(50, 200, 50, 255), // Green
            model_matrix: nalgebra::Matrix4::identity(),
        }));

        // Add 3D sphere
        scene.add(Box::new(Sphere3D {
            center: Point3::new(0.0, 0.0, 0.0),
            radius: 0.5,
            color: Color::new(50, 100, 255, 255), // Blue
            model_matrix: nalgebra::Matrix4::identity(),
        }));

        let mut timeline = Timeline::new(scene, ctx);
        // Play an animation on the 3D object to test Transform!
        timeline.play(Rotate::new(arrow_ref, Vector3::z(), Point3::origin(), 60));

        // Skip video writing for this test to speed it up and avoid pipeline dependencies.
        // We just run through the frames to ensure rendering runs without crashing.
        timeline.render(|_ctx| {
            // Frame is rendered, WGPU ran!
        });
    }
    #[test]
    fn test_arrow_perspective() {
        let width = 1920u32;
        let height = 1080u32;

        let ctx = Context {
            pixmap: Pixmap::new(width, height).unwrap(),
            scene_config: SceneConfig {
                width: 16.0,
                height: 9.0,
                output_width: width,
                output_height: height,
                scale_factor: height as GMFloat / 16.0,
            },
        };

        let mut scene = Scene::default();
        // Camera at z=1.0, very close!
        scene.camera.position = Point3::new(0.0, 0.0, 1.0);
        scene.camera.set_look_at(Vector3::new(0.0, 0.0, -1.0));

        let arrow = scene.add(Box::new(Arrow3D {
            start: Point3::new(0.0, 0.0, 0.0),
            end: Point3::new(1.0, 1.0, 0.0),
            shaft_radius: 0.02,
            head_radius: 0.05,
            head_length: 0.1,
            color: Color::new(255, 50, 50, 255),
            model_matrix: nalgebra::Matrix4::identity(),
        }));

        let mut timeline = Timeline::new(scene, ctx);
        // Only render 1 frame to test perspective
        timeline.render(|_ctx| {});
    }

    #[test]
    fn bench_scene_3d_encode() {
        let width = 1920u32;
        let height = 1080u32;

        let ctx = Context {
            scene_config: SceneConfig {
                width: 16.0,
                height: 9.0,
                output_width: width,
                output_height: height,
                scale_factor: height as GMFloat / 16.0,
            },
        };

        let mut scene = Scene::default();
        scene.camera.position = Point3::new(0.0, 0.0, 2.0);
        scene.camera.set_look_at(Vector3::new(0.0, 0.0, -1.0));

        // Generate 500 spheres for heavy load using Rasterization!
        for i in 0..500 {
            let x = (i as f32 % 10.0) - 5.0 + 0.5;
            let y = ((i / 10) as f32 % 10.0) - 5.0 + 0.5;
            let z = ((i / 100) as f32 % 10.0) - 5.0;
            scene.add(Box::new(
                crate::mobjects::mesh_3d::TriangleMesh3D::uv_sphere(
                    Point3::new(x as GMFloat, y as GMFloat, z as GMFloat),
                    0.4,
                    24, // segments
                    24, // rings
                    Color::new(200, 100, 50, 255),
                ),
            ));
        }

        struct RotateCamera {
            frames: u32,
        }
        impl crate::animation::Animation for RotateCamera {
            fn update(&mut self, alpha: GMFloat, scene: &mut Scene) {
                let angle = alpha as f32 * std::f32::consts::PI * 2.0;
                let center_z = -3.0;
                let radius = 5.0;
                let cam_x = angle.sin() * radius;
                let cam_z = center_z + angle.cos() * radius;
                scene.camera.position =
                    nalgebra::Point3::new(cam_x as GMFloat, 0.0, cam_z as GMFloat);
                scene.camera.set_look_at(nalgebra::Vector3::new(
                    -cam_x as GMFloat,
                    0.0,
                    (center_z - cam_z) as GMFloat,
                ));
            }
            fn total_frames(&self) -> u32 {
                self.frames
            }
        }

        let mut timeline = Timeline::new(scene, ctx);
        timeline.play(RotateCamera { frames: 300 });

        let video_config = VideoConfig {
            filename: "bench_core_nv12.mp4".to_owned(),
            framerate: 60,
            output_width: width,
            output_height: height,
            color_order: ColorOrder::Nv12,
        };

        let mut video_backend = VideoBackend {
            backend_type: crate::video_backend::VideoBackendType::FfmpegPipe(
                crate::video_backend::FfmpegPipeBackend::new(
                    &video_config,
                    crate::video_backend::FfmpegPipeEncoder::Libx264,
                    false,
                ),
            ),
        };

        let start = std::time::Instant::now();
        while timeline.step_frame() {
            let mut buf = video_backend.acquire_buffer();
            if let Some(nv12) = timeline.nv12_image_bytes() {
                buf.as_mut_slice().copy_from_slice(nv12);
            } else {
                println!("Frame NV12 bytes is None!");
                buf.as_mut_slice().fill(0);
            }
            video_backend.submit_frame(buf);
        }
        video_backend.close().unwrap();

        println!(
            "Rust Core 3D Encode time for 180 frames: {:?}",
            start.elapsed()
        );
    }
}
