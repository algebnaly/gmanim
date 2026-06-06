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
    fn as_primitive_data(&self) -> crate::wgpu::renderer::PrimitiveData3D;
}

// ═══════════════════════════════════════════════════════════════════════════
// 3D Scene (The Ray Marcher)
// ═══════════════════════════════════════════════════════════════════════════

/// A 3D scene rendered via CPU Ray Marching.
/// It implements `Mobject` so it can be added to the 2D Timeline seamlessly.
pub struct Scene3D {
    pub camera: Camera,
    pub objects: Vec<Box<dyn Object3D>>,
    pub light_pos: Point3<GMFloat>,
    pub light_color: Vector3<GMFloat>,
    pub ambient: GMFloat,
    pub max_steps: usize,
    pub max_dist: GMFloat,
    pub surf_dist: GMFloat,
    pub model_matrix: nalgebra::Matrix4<GMFloat>,
}

impl Default for Scene3D {
    fn default() -> Self {
        Self {
            camera: Camera::default(),
            objects: vec![],
            light_pos: Point3::new(10.0, 10.0, 10.0),
            light_color: Vector3::new(1.0, 1.0, 1.0),
            ambient: 0.2,
            max_steps: 100,
            max_dist: 100.0,
            surf_dist: 0.001,
            model_matrix: nalgebra::Matrix4::identity(),
        }
    }
}

impl Scene3D {
    pub fn add(&mut self, obj: Box<dyn Object3D>) {
        self.objects.push(obj);
    }

    /// Evaluates the entire scene's 3D (union of all objects)
    fn scene_3d(&self, p: &Point3<GMFloat>) -> (GMFloat, Color) {
        let mut min_dist = self.max_dist;
        let mut min_color = Color::default();
        for obj in &self.objects {
            let d = obj.distance(p);
            if d < min_dist {
                min_dist = d;
                min_color = obj.color(p);
            }
        }
        (min_dist, min_color)
    }

    /// Calculates the normal using finite differences of the 3D
    fn calculate_normal(&self, p: &Point3<GMFloat>) -> Vector3<GMFloat> {
        let e = 0.001;
        let dx = self.scene_3d(&Point3::new(p.x + e, p.y, p.z)).0
            - self.scene_3d(&Point3::new(p.x - e, p.y, p.z)).0;
        let dy = self.scene_3d(&Point3::new(p.x, p.y + e, p.z)).0
            - self.scene_3d(&Point3::new(p.x, p.y - e, p.z)).0;
        let dz = self.scene_3d(&Point3::new(p.x, p.y, p.z + e)).0
            - self.scene_3d(&Point3::new(p.x, p.y, p.z - e)).0;
        Vector3::new(dx, dy, dz).normalize()
    }

    /// Calculates soft shadows
    fn calculate_shadow(&self, ro: &Point3<GMFloat>, rd: &Vector3<GMFloat>) -> GMFloat {
        let mut res: GMFloat = 1.0;
        let mut t: GMFloat = 0.05; // start slightly offset to avoid self-intersection
        for _ in 0..self.max_steps {
            let p = ro + rd * t;
            let (d, _) = self.scene_3d(&p);
            if d < self.surf_dist {
                return 0.0; // fully shadowed
            }
            res = res.min(10.0 * d / t);
            t += d;
            if t > self.max_dist {
                break;
            }
        }
        res.clamp(0.0 as GMFloat, 1.0 as GMFloat)
    }

    /// Blinn-Phong Shading
    fn shade(
        &self,
        hit_point: &Point3<GMFloat>,
        normal: &Vector3<GMFloat>,
        object_color: Color,
    ) -> Color {
        let obj_col_vec = Vector3::new(
            object_color.r as GMFloat / 255.0,
            object_color.g as GMFloat / 255.0,
            object_color.b as GMFloat / 255.0,
        );

        let light_dir = (self.light_pos - hit_point).normalize();
        let diffuse = normal.dot(&light_dir).max(0.0);

        let view_dir = (self.camera.position - hit_point).normalize();
        let half_dir = (light_dir + view_dir).normalize();
        let specular = normal.dot(&half_dir).max(0.0).powf(32.0);

        let shadow = self.calculate_shadow(&(hit_point + normal * 0.01), &light_dir);

        let final_intensity = self.ambient + diffuse * shadow;
        let final_col = obj_col_vec * final_intensity + self.light_color * specular * shadow;

        Color::new(
            (final_col.x.clamp(0.0, 1.0) * 255.0) as u8,
            (final_col.y.clamp(0.0, 1.0) * 255.0) as u8,
            (final_col.z.clamp(0.0, 1.0) * 255.0) as u8,
            object_color.a,
        )
    }

    fn render_ray(&self, ro: &Point3<GMFloat>, rd: &Vector3<GMFloat>) -> Option<Color> {
        let mut t = 0.0;
        let mut hit_color = Color::default();

        for _ in 0..self.max_steps {
            let p = ro + rd * t;
            let (d, col) = self.scene_3d(&p);

            if d < self.surf_dist {
                let hit_point = ro + rd * t;
                let normal = self.calculate_normal(&hit_point);
                return Some(self.shade(&hit_point, &normal, col));
            }
            t += d;
            if t > self.max_dist {
                break;
            }
        }
        None
    }
}

impl Draw for Scene3D {
    fn draw(&self, ctx: &mut Context, _parent_matrix: nalgebra::Matrix4<GMFloat>) {
        let width = ctx.scene_config.output_width as usize;
        let height = ctx.scene_config.output_height as usize;
        let pixels = ctx.pixmap.pixels_mut();

        // 增加抗锯齿采样率（默认 2x2 = 4 次采样，极大提升边缘清晰度）
        let aa = 2;

        // Parallel Ray Marching across all pixels
        pixels.par_iter_mut().enumerate().for_each(|(i, pixel)| {
            let x = i % width;
            let y = i / width;

            let mut r = 0.0;
            let mut g = 0.0;
            let mut b = 0.0;
            let mut hits = 0;

            for dy in 0..aa {
                for dx in 0..aa {
                    // 在像素内部进行亚像素偏移
                    let offset_x = (dx as f32 + 0.5) / aa as f32 - 0.5;
                    let offset_y = (dy as f32 + 0.5) / aa as f32 - 0.5;

                    let (ro, rd) = self.camera.get_ray(
                        x as f32 + offset_x,
                        y as f32 + offset_y,
                        width as f32,
                        height as f32,
                    );

                    if let Some(col) = self.render_ray(&ro, &rd) {
                        r += col.r as f32;
                        g += col.g as f32;
                        b += col.b as f32;
                        hits += 1;
                    }
                }
            }

            if hits > 0 {
                // 如果只遮盖了一部分亚像素，可以和背景进行 Alpha 混合实现边缘抗锯齿
                let alpha = (hits as f32 / (aa * aa) as f32) * 255.0;
                let final_r = (r / hits as f32) as u8;
                let final_g = (g / hits as f32) as u8;
                let final_b = (b / hits as f32) as u8;

                *pixel = tiny_skia::PremultipliedColorU8::from_rgba(
                    final_r,
                    final_g,
                    final_b,
                    alpha as u8,
                )
                .unwrap_or(*pixel);
            }
        });
    }
}

impl Transform for Scene3D {
    fn get_model_matrix(&self) -> nalgebra::Matrix4<GMFloat> {
        self.model_matrix
    }
    fn set_model_matrix(&mut self, mat: nalgebra::Matrix4<GMFloat>) {
        self.model_matrix = mat;
    }
}

impl Mobject for Scene3D {}

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
    fn as_primitive_data(&self) -> crate::wgpu::renderer::PrimitiveData3D {
        crate::wgpu::renderer::PrimitiveData3D {
            color: [
                self.color.r as f32 / 255.0,
                self.color.g as f32 / 255.0,
                self.color.b as f32 / 255.0,
                self.color.a as f32 / 255.0,
            ],
            params: [
                self.center.x as f32,
                self.center.y as f32,
                self.center.z as f32,
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
    fn as_primitive_data(&self) -> crate::wgpu::renderer::PrimitiveData3D {
        crate::wgpu::renderer::PrimitiveData3D {
            color: [
                self.color.r as f32 / 255.0,
                self.color.g as f32 / 255.0,
                self.color.b as f32 / 255.0,
                self.color.a as f32 / 255.0,
            ],
            params: [
                self.a.x as f32,
                self.a.y as f32,
                self.a.z as f32,
                self.b.x as f32,
                self.b.y as f32,
                self.b.z as f32,
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

    fn as_primitive_data(&self) -> crate::wgpu::renderer::PrimitiveData3D {
        crate::wgpu::renderer::PrimitiveData3D {
            color: [
                self.color.r as f32 / 255.0,
                self.color.g as f32 / 255.0,
                self.color.b as f32 / 255.0,
                self.color.a as f32 / 255.0,
            ],
            params: [
                self.start.x as f32,
                self.start.y as f32,
                self.start.z as f32,
                self.end.x as f32,
                self.end.y as f32,
                self.end.z as f32,
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

pub struct Box3D {
    pub center: Point3<GMFloat>,
    pub size: Vector3<GMFloat>,
    pub x_axis: Vector3<GMFloat>,
    pub y_axis: Vector3<GMFloat>,
    pub z_axis: Vector3<GMFloat>,
    pub color: Color,
    pub model_matrix: nalgebra::Matrix4<GMFloat>,
}

impl Object3D for Box3D {
    fn distance(&self, p: &Point3<GMFloat>) -> GMFloat {
        let pt = p - self.center;
        let local_p = Vector3::new(pt.dot(&self.x_axis), pt.dot(&self.y_axis), pt.dot(&self.z_axis));
        let d = Vector3::new(local_p.x.abs() - self.size.x, local_p.y.abs() - self.size.y, local_p.z.abs() - self.size.z);
        
        let d_max = Vector3::new(d.x.max(0.0), d.y.max(0.0), d.z.max(0.0));
        let max_comp = d.x.max(d.y).max(d.z).min(0.0);
        d_max.norm() + max_comp
    }
    
    fn color(&self, _p: &Point3<GMFloat>) -> Color {
        self.color
    }
    
    fn as_primitive_data(&self) -> crate::wgpu::renderer::PrimitiveData3D {
        crate::wgpu::renderer::PrimitiveData3D {
            color: [
                self.color.r as f32 / 255.0,
                self.color.g as f32 / 255.0,
                self.color.b as f32 / 255.0,
                self.color.a as f32 / 255.0,
            ],
            params: [
                self.center.x as f32,
                self.center.y as f32,
                self.center.z as f32,
                self.size.x as f32,
                self.size.y as f32,
                self.size.z as f32,
                self.x_axis.x as f32,
                self.x_axis.y as f32,
                self.x_axis.z as f32,
                self.y_axis.x as f32,
                self.y_axis.y as f32,
                self.y_axis.z as f32,
            ],
            shape_type: 3,
            padding: [0; 3],
        }
    }
}

impl Transform for Box3D {
    fn get_model_matrix(&self) -> nalgebra::Matrix4<GMFloat> {
        self.model_matrix
    }
    fn set_model_matrix(&mut self, mat: nalgebra::Matrix4<GMFloat>) {
        self.model_matrix = mat;
    }
}

impl Draw for Box3D {
    fn draw(&self, _ctx: &mut Context, _parent_matrix: nalgebra::Matrix4<GMFloat>) {}
}

impl Mobject for Box3D {
    fn as_3d(&self) -> Option<&dyn crate::mobjects::object_3d::Object3D> {
        Some(self)
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
    use crate::video_backend::VideoBackend;
    use crate::video_backend::{vaapi::FfmpegVaapiBackend, ColorOrder, VideoConfig};
    use crate::{Context, Scene, SceneConfig};
    use tiny_skia::Pixmap;

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
        }));

        // Add 3D Line segment
        scene.add(Box::new(LineSegment3D {
            a: Point3::new(-2.0, -1.0, -1.0),
            b: Point3::new(1.0, -2.0, 1.0),
            radius: 0.15,
            color: Color::new(50, 200, 50, 255), // Green
        }));

        // Add 3D sphere
        scene.add(Box::new(Sphere3D {
            center: Point3::new(0.0, 0.0, 0.0),
            radius: 0.5,
            color: Color::new(50, 100, 255, 255), // Blue
        }));

        let mut timeline = Timeline::new(scene, ctx);
        // Play an animation on the 3D object to test Transform!
        timeline.play(Rotate::new(arrow_ref, Vector3::z(), Point3::origin(), 60));

        // Skip video writing for this test to speed it up and avoid pipeline dependencies.
        // We just run through the frames to ensure rendering runs without crashing.
        timeline.render(|_ctx| {
            // Frame is rendered, WGPU ran!
        });
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
        }));

        let mut timeline = Timeline::new(scene, ctx);
        // Only render 1 frame to test perspective
        timeline.render(|_ctx| {});
    }
}

}
