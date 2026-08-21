use std::collections::{HashMap, HashSet};

use crate::mobjects::mesh_3d::{AlphaMode3D, SurfaceMaterial, Vertex};
use crate::mobjects::object_3d::SdfPrimitive;
use crate::mobjects::{Rectangle, RectangleId};

use super::mesh_2d::{
    CachedRectangle2D, Instance2D, Mesh2DBatch, Mesh2DSubmission, build_ordered_mesh_2d_batches,
    rectangle_analytic_aa_params,
};
use super::{CameraUniform, CameraUniform2D, Mesh3DDraw};

pub(super) struct PreparedSdfPrimitive {
    pub(super) primitive: SdfPrimitive,
    pub(super) material_index: u32,
}

pub(super) struct PreparedScene {
    pub(super) camera_uniform: CameraUniform,
    pub(super) camera_uniform_2d: CameraUniform2D,
    pub(super) sdf_primitives: Vec<PreparedSdfPrimitive>,
    pub(super) surface_materials: Vec<SurfaceMaterial>,
    pub(super) mesh_vertices: Vec<Vertex>,
    pub(super) mesh_indices: Vec<u32>,
    pub(super) mesh_draws_3d: Vec<Mesh3DDraw>,
    pub(super) mesh_batches_2d: Vec<Mesh2DBatch>,
    pub(super) background_color: [f32; 4],
}

#[derive(Default)]
pub(super) struct ScenePreparer {
    rectangle_cache_2d: HashMap<RectangleId, CachedRectangle2D>,
}

impl ScenePreparer {
    pub(super) fn prepare(
        &mut self,
        scene: &crate::Scene,
        scene_config: &crate::SceneConfig,
        ssaa_factor: u32,
    ) -> PreparedScene {
        let output_w = scene_config.output_width as f32;
        let output_h = scene_config.output_height as f32;

        let bg = scene.background_color.to_array();
        let bg_clear_color = [bg[0] as f32, bg[1] as f32, bg[2] as f32, bg[3] as f32];

        let (has_clip, clip_x, clip_y, clip_w, clip_h) = match scene.clip_rect {
            Some(crate::ClipRect::Pixel(x, y, w, h)) => {
                (true, x as f32, y as f32, w as f32, h as f32)
            }
            Some(crate::ClipRect::Logical(cx, cy, w, h)) => {
                let (o_left, o_right, o_bottom, o_top, _, _) = scene.camera.ortho_params();
                let log_w = o_right - o_left;
                let log_h = o_top - o_bottom;

                let tl_x = cx - w / 2.0;
                let tl_y = cy + h / 2.0;

                let norm_x = (tl_x - o_left) / log_w;
                let norm_y = (o_top - tl_y) / log_h;
                let norm_w = w / log_w;
                let norm_h = h / log_h;

                (
                    true,
                    norm_x * output_w,
                    norm_y * output_h,
                    norm_w * output_w,
                    norm_h * output_h,
                )
            }
            None => (false, 0.0, 0.0, 0.0, 0.0),
        };

        let mut sdf_primitives = Vec::new();
        let mut mesh_vertices = Vec::new();
        let mut mesh_indices = Vec::new();
        let mut mesh_draws_3d = Vec::new();
        let mut surface_materials = Vec::new();

        let mut mesh_submissions_2d = Vec::new();
        let mut active_rectangles_2d = HashSet::new();

        struct VulkanDataCollector<'a> {
            sdf_primitives: &'a mut Vec<PreparedSdfPrimitive>,
            mesh_vertices: &'a mut Vec<Vertex>,
            mesh_indices: &'a mut Vec<u32>,
            mesh_draws_3d: &'a mut Vec<Mesh3DDraw>,
            surface_materials: &'a mut Vec<SurfaceMaterial>,
            mesh_submissions_2d: &'a mut Vec<Mesh2DSubmission>,
            rectangle_cache_2d: &'a mut HashMap<RectangleId, CachedRectangle2D>,
            active_rectangles_2d: &'a mut HashSet<RectangleId>,
            camera_position: nalgebra::Point3<crate::GMFloat>,
            camera_look: nalgebra::Vector3<crate::GMFloat>,
        }

        impl<'a> crate::mobjects::RenderVisitor for VulkanDataCollector<'a> {
            fn push_mesh_2d(
                &mut self,
                mesh: &crate::mobjects::mesh_2d::TriangleMesh2D,
                transform: nalgebra::Matrix4<crate::GMFloat>,
            ) {
                self.mesh_submissions_2d.push(Mesh2DSubmission {
                    geometry: mesh.geometry(),
                    instance: Instance2D::new(
                        transform,
                        mesh.color(),
                        [0.0, 0.0, mesh.aa_mode, 0.0],
                    ),
                    dynamic: false,
                });
            }

            fn push_rectangle_2d(
                &mut self,
                id: RectangleId,
                rectangle: &Rectangle,
                geometry_revision: u64,
                dynamic: bool,
                transform: nalgebra::Matrix4<crate::GMFloat>,
            ) {
                self.active_rectangles_2d.insert(id);
                let rebuild = self
                    .rectangle_cache_2d
                    .get(&id)
                    .map(|cached| {
                        cached.geometry_revision != geometry_revision
                            || !cached.source.same_geometry(rectangle)
                    })
                    .unwrap_or(true);
                if rebuild {
                    self.rectangle_cache_2d.insert(
                        id,
                        CachedRectangle2D {
                            geometry_revision,
                            source: rectangle.clone(),
                            geometry: rectangle.tessellate().geometry(),
                        },
                    );
                }
                let geometry = self.rectangle_cache_2d[&id].geometry.clone();
                self.mesh_submissions_2d.push(Mesh2DSubmission {
                    geometry,
                    instance: Instance2D::new(
                        transform,
                        [
                            rectangle.color.r as f32 / 255.0,
                            rectangle.color.g as f32 / 255.0,
                            rectangle.color.b as f32 / 255.0,
                            rectangle.color.a as f32 / 255.0,
                        ],
                        rectangle_analytic_aa_params(rectangle),
                    ),
                    dynamic,
                });
            }

            fn push_surface_3d(&mut self, surface: crate::mobjects::Surface3DSubmission<'_>) {
                let material_index = self.surface_materials.len() as u32;
                self.surface_materials.push(surface.material);
                match surface.geometry {
                    crate::mobjects::Geometry3DRef::Mesh(mesh) => {
                        let base_index = self.mesh_vertices.len() as u32;
                        let first_index = self.mesh_indices.len() as u32;
                        let mut world_center = nalgebra::Point3::origin();
                        for vertex in &mesh.vertices {
                            let position = nalgebra::Point3::new(
                                vertex.position[0] as crate::GMFloat,
                                vertex.position[1] as crate::GMFloat,
                                vertex.position[2] as crate::GMFloat,
                            );
                            let world_position = surface.transform.transform_point(&position);
                            world_center.coords += world_position.coords;
                            let normal = nalgebra::Vector3::new(
                                vertex.normal[0] as crate::GMFloat,
                                vertex.normal[1] as crate::GMFloat,
                                vertex.normal[2] as crate::GMFloat,
                            );
                            let world_normal =
                                surface.transform.transform_vector(&normal).normalize();
                            self.mesh_vertices.push(Vertex {
                                position: [
                                    world_position.x as f32,
                                    world_position.y as f32,
                                    world_position.z as f32,
                                ],
                                normal: [
                                    world_normal.x as f32,
                                    world_normal.y as f32,
                                    world_normal.z as f32,
                                ],
                                color: vertex.color,
                                surface_coord: vertex.surface_coord,
                            });
                        }
                        for index in &mesh.indices {
                            self.mesh_indices.push(*index + base_index);
                        }
                        if !mesh.vertices.is_empty() && !mesh.indices.is_empty() {
                            world_center.coords /= mesh.vertices.len() as crate::GMFloat;
                            self.mesh_draws_3d.push(Mesh3DDraw {
                                first_index,
                                index_count: mesh.indices.len() as u32,
                                material_index,
                                transparent: matches!(
                                    surface.material.alpha_mode,
                                    AlphaMode3D::Blend(_)
                                ),
                                view_depth: (world_center - self.camera_position)
                                    .dot(&self.camera_look)
                                    as f32,
                            });
                        }
                    }
                    crate::mobjects::Geometry3DRef::Sdf(sdf) => {
                        assert!(
                            matches!(surface.material.alpha_mode, AlphaMode3D::Opaque),
                            "transparent SDF surfaces require entry/exit ray marching"
                        );
                        self.sdf_primitives.push(PreparedSdfPrimitive {
                            primitive: sdf.transformed_primitive(surface.transform),
                            material_index,
                        });
                    }
                }
            }
        }

        let mut collector = VulkanDataCollector {
            sdf_primitives: &mut sdf_primitives,
            mesh_vertices: &mut mesh_vertices,
            mesh_indices: &mut mesh_indices,
            mesh_draws_3d: &mut mesh_draws_3d,
            surface_materials: &mut surface_materials,
            mesh_submissions_2d: &mut mesh_submissions_2d,
            rectangle_cache_2d: &mut self.rectangle_cache_2d,
            active_rectangles_2d: &mut active_rectangles_2d,
            camera_position: scene.camera.position,
            camera_look: scene.camera.look_at_dir(),
        };

        scene.world.submit_to_renderer(&mut collector);
        drop(collector);
        self.rectangle_cache_2d
            .retain(|id, _| active_rectangles_2d.contains(id));
        let mesh_batches_2d = build_ordered_mesh_2d_batches(mesh_submissions_2d);

        let camera_uniform_2d = CameraUniform2D {
            width: output_w,
            height: output_h,
            scale_factor: scene_config.scale_factor as f32,
            _pad: 0.0,
        };

        let look = scene.camera.look_at_dir();
        let camera_uniform = CameraUniform {
            pos: [
                scene.camera.position.x as f32,
                scene.camera.position.y as f32,
                scene.camera.position.z as f32,
            ],
            _padding0: 0,
            look_at: [look.x as f32, look.y as f32, look.z as f32],
            _padding1: 0,
            up: [
                scene.camera.up_dir().x as f32,
                scene.camera.up_dir().y as f32,
                scene.camera.up_dir().z as f32,
            ],
            fov: scene.camera.fov() as f32,
            width: output_w,
            height: output_h,
            proj_type: scene.camera.proj_type(),
            ortho_left: scene.camera.ortho_params().0 as f32,
            ortho_right: scene.camera.ortho_params().1 as f32,
            ortho_bottom: scene.camera.ortho_params().2 as f32,
            ortho_top: scene.camera.ortho_params().3 as f32,
            has_clip: if has_clip { 1 } else { 0 },
            clip_x,
            clip_y,
            clip_w,
            clip_h,
            aa_level: scene.aa_level,
            num_primitives: sdf_primitives.len() as u32,
            raster_scale: ssaa_factor,
            has_raster_surfaces: mesh_draws_3d.iter().any(|draw| !draw.is_transparent()) as u32,
            proj_mat: {
                if scene.camera.proj_type() == 0 {
                    crate::camera::Projection::perspective_wgpu(
                        scene.camera.fov() as f32,
                        output_w / output_h,
                        scene.camera.perspective_params().0 as f32,
                        scene.camera.perspective_params().1 as f32,
                    )
                } else {
                    let aspect = output_w / output_h;
                    let ortho_params = scene.camera.ortho_params();
                    // ortho_params returns (left, right, bottom, top, near, far) where left/right are often without aspect ratio applied
                    // Actually, let's just use the exact params from the camera
                    crate::camera::Projection::orthographic_wgpu(
                        ortho_params.0 as f32,
                        ortho_params.1 as f32,
                        ortho_params.2 as f32,
                        ortho_params.3 as f32,
                        ortho_params.4 as f32,
                        ortho_params.5 as f32,
                    )
                }
            },
            light_pos: [
                scene.point_light.position.x as f32,
                scene.point_light.position.y as f32,
                scene.point_light.position.z as f32,
            ],
            light_intensity: scene.point_light.intensity as f32,
            light_color: [
                scene.point_light.color.r as f32 / 255.0,
                scene.point_light.color.g as f32 / 255.0,
                scene.point_light.color.b as f32 / 255.0,
            ],
            environment_intensity: scene.environment_light.intensity as f32,
            environment_color: [
                scene.environment_light.color.r as f32 / 255.0,
                scene.environment_light.color.g as f32 / 255.0,
                scene.environment_light.color.b as f32 / 255.0,
            ],
            environment_rotation: scene.environment_light.rotation_radians as f32,
            background_color: bg_clear_color,
        };

        mesh_draws_3d.sort_by(|left, right| {
            left.is_transparent()
                .cmp(&right.is_transparent())
                .then_with(|| {
                    right
                        .view_depth
                        .partial_cmp(&left.view_depth)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });

        PreparedScene {
            camera_uniform,
            camera_uniform_2d,
            sdf_primitives,
            surface_materials,
            mesh_vertices,
            mesh_indices,
            mesh_draws_3d,
            mesh_batches_2d,
            background_color: bg_clear_color,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use nalgebra::{Point3, Vector3};

    use super::ScenePreparer;
    use crate::mobjects::Rectangle;
    use crate::mobjects::mesh_3d::{SurfaceMaterial, TriangleMesh3D};
    use crate::mobjects::object_3d::{SdfPrimitive, Sphere3D};
    use crate::{Color, Scene, SceneConfig};

    #[test]
    fn preparation_collects_scene_without_a_vulkan_device() {
        let mut scene = Scene::default();
        scene.add(Sphere3D {
            radius: 1.25,
            material: SurfaceMaterial::default(),
        });
        scene.add(TriangleMesh3D::box_mesh(
            Point3::new(0.0, 0.0, -2.0),
            Vector3::new(0.5, 0.5, 0.5),
            Color::white(),
        ));
        scene.add_rectangle(Rectangle {
            p0: Point3::new(-1.0, -1.0, 0.0),
            p1: Point3::new(1.0, -1.0, 0.0),
            p2: Point3::new(1.0, 1.0, 0.0),
            p3: Point3::new(-1.0, 1.0, 0.0),
            ..Default::default()
        });

        let config = SceneConfig::default();
        let mut preparer = ScenePreparer::default();
        let first = preparer.prepare(&scene, &config, 1);

        assert_eq!(first.sdf_primitives.len(), 1);
        assert_eq!(first.camera_uniform.num_primitives, 1);
        assert_eq!(first.surface_materials.len(), 2);
        assert!(!first.mesh_vertices.is_empty());
        assert!(!first.mesh_indices.is_empty());
        assert_eq!(first.mesh_draws_3d.len(), 1);
        assert_eq!(first.mesh_batches_2d.len(), 1);
        assert!(matches!(
            first.sdf_primitives[0].primitive,
            SdfPrimitive::Sphere { radius, .. } if radius == 1.25
        ));

        let second = preparer.prepare(&scene, &config, 1);
        assert!(Arc::ptr_eq(
            &first.mesh_batches_2d[0].geometry,
            &second.mesh_batches_2d[0].geometry,
        ));
    }
}
