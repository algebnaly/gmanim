use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::mobjects::GridStyle3D;
use crate::mobjects::mesh_3d::{AlphaMode3D, Geometry3DId, SurfaceMaterial, Vertex};
use crate::mobjects::object_3d::SdfPrimitive;

use super::mesh_2d::{
    Instance2D, Mesh2DBatch, Mesh2DSubmission, build_ordered_mesh_2d_batches,
};
use super::{CameraUniform, CameraUniform2D, Mesh3DDraw};

type TransformBits = [[u32; 4]; 4];
type MeshCacheKey3D = (Geometry3DId, TransformBits);

struct CachedMesh3D {
    vertices: Arc<[Vertex]>,
    world_center: nalgebra::Point3<crate::GMFloat>,
}

fn matrix_to_bits(matrix: &nalgebra::Matrix4<crate::GMFloat>) -> TransformBits {
    std::array::from_fn(|row| std::array::from_fn(|col| matrix[(row, col)].to_bits()))
}

pub(super) struct PreparedSdfPrimitive {
    pub(super) primitive: SdfPrimitive,
    pub(super) material_index: u32,
}

pub(super) struct PreparedGrid3D {
    pub(super) origin: nalgebra::Point3<crate::GMFloat>,
    pub(super) u_axis: nalgebra::Vector3<crate::GMFloat>,
    pub(super) v_axis: nalgebra::Vector3<crate::GMFloat>,
    pub(super) half_extent: crate::GMFloat,
    pub(super) style: GridStyle3D,
}

pub(super) struct PreparedScene {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) camera_uniform: CameraUniform,
    pub(super) camera_uniform_2d: CameraUniform2D,
    pub(super) sdf_primitives: Vec<PreparedSdfPrimitive>,
    pub(super) surface_materials: Vec<SurfaceMaterial>,
    pub(super) mesh_vertices: Vec<Vertex>,
    pub(super) mesh_indices: Vec<u32>,
    pub(super) mesh_draws_3d: Vec<Mesh3DDraw>,
    pub(super) grids_3d: Vec<PreparedGrid3D>,
    pub(super) mesh_batches_2d: Vec<Mesh2DBatch>,
    pub(super) background_color: [f32; 4],
}

#[derive(Default)]
pub(super) struct ScenePreparer {
    mesh_cache_3d: HashMap<MeshCacheKey3D, CachedMesh3D>,
}

/// Opaque draws are ordered front-to-back so that early depth rejection
/// avoids shading pixels that are later covered by a nearer mesh.
/// Transparent draws stay back-to-front because they are alpha-composited.
fn sort_mesh_draws_3d(draws: &mut [Mesh3DDraw]) {
    draws.sort_by(
        |left, right| match (left.is_transparent(), right.is_transparent()) {
            (false, true) => std::cmp::Ordering::Less,
            (true, false) => std::cmp::Ordering::Greater,
            (false, false) => left
                .view_depth
                .partial_cmp(&right.view_depth)
                .unwrap_or(std::cmp::Ordering::Equal),
            (true, true) => right
                .view_depth
                .partial_cmp(&left.view_depth)
                .unwrap_or(std::cmp::Ordering::Equal),
        },
    );
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
        let bg_clear_color = [bg[0], bg[1], bg[2], bg[3]];

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
        let mut grids_3d = Vec::new();

        let mut mesh_submissions_2d = Vec::new();
        let mut active_meshes_3d = HashSet::new();

        struct VulkanDataCollector<'a> {
            sdf_primitives: &'a mut Vec<PreparedSdfPrimitive>,
            mesh_vertices: &'a mut Vec<Vertex>,
            mesh_indices: &'a mut Vec<u32>,
            mesh_draws_3d: &'a mut Vec<Mesh3DDraw>,
            surface_materials: &'a mut Vec<SurfaceMaterial>,
            grids_3d: &'a mut Vec<PreparedGrid3D>,
            mesh_submissions_2d: &'a mut Vec<Mesh2DSubmission>,
            mesh_cache_3d: &'a mut HashMap<MeshCacheKey3D, CachedMesh3D>,
            active_meshes_3d: &'a mut HashSet<MeshCacheKey3D>,
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
                    instance: Instance2D::new(transform, mesh.color(), mesh.aa_params()),
                    dynamic: mesh.is_dynamic(),
                });
            }

            fn push_surface_3d(&mut self, surface: crate::mobjects::Surface3DSubmission<'_>) {
                let material_index = self.surface_materials.len() as u32;
                self.surface_materials.push(surface.material);
                match surface.geometry {
                    crate::mobjects::Geometry3DRef::Mesh(mesh) => {
                        let vertices = mesh.vertices();
                        let indices = mesh.indices();
                        if vertices.is_empty() || indices.is_empty() {
                            return;
                        }
                        let cache_key = (mesh.geometry().id(), matrix_to_bits(&surface.transform));
                        self.active_meshes_3d.insert(cache_key);
                        let cached = self.mesh_cache_3d.entry(cache_key).or_insert_with(|| {
                            let linear = surface.transform.fixed_view::<3, 3>(0, 0).into_owned();
                            let normal_matrix = linear
                                .try_inverse()
                                .map(|inverse| inverse.transpose())
                                .unwrap_or_else(|| {
                                    // Cofactors preserve normals for surfaces flattened to a plane.
                                    // Fully collapsed surfaces have no geometric normal.
                                    nalgebra::Matrix3::from_columns(&[
                                        linear.column(1).cross(&linear.column(2)),
                                        linear.column(2).cross(&linear.column(0)),
                                        linear.column(0).cross(&linear.column(1)),
                                    ])
                                });
                            let mut world_center = nalgebra::Point3::origin();
                            let transformed: Vec<Vertex> = vertices
                                .iter()
                                .map(|vertex| {
                                    let position = nalgebra::Point3::new(
                                        vertex.position[0] as crate::GMFloat,
                                        vertex.position[1] as crate::GMFloat,
                                        vertex.position[2] as crate::GMFloat,
                                    );
                                    let world_position =
                                        surface.transform.transform_point(&position);
                                    world_center.coords += world_position.coords;
                                    let normal = nalgebra::Vector3::new(
                                        vertex.normal[0] as crate::GMFloat,
                                        vertex.normal[1] as crate::GMFloat,
                                        vertex.normal[2] as crate::GMFloat,
                                    );
                                    let world_normal = (normal_matrix * normal)
                                        .try_normalize(0.0)
                                        .filter(|normal| {
                                            normal.iter().all(|value| value.is_finite())
                                        })
                                        .unwrap_or_else(nalgebra::Vector3::z);
                                    Vertex {
                                        position: [
                                            world_position.x,
                                            world_position.y,
                                            world_position.z,
                                        ],
                                        normal: [world_normal.x, world_normal.y, world_normal.z],
                                        color: vertex.color,
                                        surface_coord: vertex.surface_coord,
                                    }
                                })
                                .collect();
                            world_center.coords /= vertices.len() as crate::GMFloat;
                            CachedMesh3D {
                                vertices: transformed.into(),
                                world_center,
                            }
                        });
                        let base_index = self.mesh_vertices.len() as u32;
                        let first_index = self.mesh_indices.len() as u32;
                        self.mesh_vertices.extend_from_slice(&cached.vertices);
                        self.mesh_indices
                            .extend(indices.iter().map(|index| *index + base_index));
                        self.mesh_draws_3d.push(Mesh3DDraw {
                            first_index,
                            index_count: indices.len() as u32,
                            material_index,
                            transparent: matches!(
                                surface.material.alpha_mode,
                                AlphaMode3D::Blend(_)
                            ),
                            view_depth: (cached.world_center - self.camera_position)
                                .dot(&self.camera_look),
                        });
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

            fn push_grid_3d(&mut self, submission: crate::mobjects::Grid3DSubmission<'_>) {
                let (u_axis, v_axis) = submission.grid.plane.axes();
                self.grids_3d.push(PreparedGrid3D {
                    origin: submission
                        .transform
                        .transform_point(&submission.grid.center),
                    u_axis: submission.transform.transform_vector(&u_axis),
                    v_axis: submission.transform.transform_vector(&v_axis),
                    half_extent: submission.grid.size * 0.5,
                    style: submission.grid.style,
                });
            }
        }

        {
            let mut collector = VulkanDataCollector {
                sdf_primitives: &mut sdf_primitives,
                mesh_vertices: &mut mesh_vertices,
                mesh_indices: &mut mesh_indices,
                mesh_draws_3d: &mut mesh_draws_3d,
                surface_materials: &mut surface_materials,
                grids_3d: &mut grids_3d,
                mesh_submissions_2d: &mut mesh_submissions_2d,
                mesh_cache_3d: &mut self.mesh_cache_3d,
                active_meshes_3d: &mut active_meshes_3d,
                camera_position: scene.camera.position,
                camera_look: scene.camera.look_at_dir(),
            };
            scene.world.submit_to_renderer(&mut collector);
        }
        self.mesh_cache_3d
            .retain(|key, _| active_meshes_3d.contains(key));
        let mesh_batches_2d = build_ordered_mesh_2d_batches(mesh_submissions_2d);

        let camera_uniform_2d = CameraUniform2D {
            width: output_w,
            height: output_h,
            scale_factor: scene_config.scale_factor,
            _pad: 0.0,
        };

        let look = scene.camera.look_at_dir();
        let camera_uniform = CameraUniform {
            pos: [
                scene.camera.position.x,
                scene.camera.position.y,
                scene.camera.position.z,
            ],
            _padding0: 0,
            look_at: [look.x, look.y, look.z],
            _padding1: 0,
            up: [
                scene.camera.up_dir().x,
                scene.camera.up_dir().y,
                scene.camera.up_dir().z,
            ],
            fov: scene.camera.fov(),
            width: output_w,
            height: output_h,
            proj_type: scene.camera.proj_type(),
            ortho_left: scene.camera.ortho_params().0,
            ortho_right: scene.camera.ortho_params().1,
            ortho_bottom: scene.camera.ortho_params().2,
            ortho_top: scene.camera.ortho_params().3,
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
                        scene.camera.fov(),
                        output_w / output_h,
                        scene.camera.perspective_params().0,
                        scene.camera.perspective_params().1,
                    )
                } else {
                    let ortho_params = scene.camera.ortho_params();
                    // ortho_params returns (left, right, bottom, top, near, far) where left/right are often without aspect ratio applied
                    // Actually, let's just use the exact params from the camera
                    crate::camera::Projection::orthographic_wgpu(
                        ortho_params.0,
                        ortho_params.1,
                        ortho_params.2,
                        ortho_params.3,
                        ortho_params.4,
                        ortho_params.5,
                    )
                }
            },
            light_pos: [
                scene.point_light.position.x,
                scene.point_light.position.y,
                scene.point_light.position.z,
            ],
            light_intensity: scene.point_light.intensity,
            light_color: [
                scene.point_light.color.r as f32 / 255.0,
                scene.point_light.color.g as f32 / 255.0,
                scene.point_light.color.b as f32 / 255.0,
            ],
            environment_intensity: scene.environment_light.intensity,
            environment_color: [
                scene.environment_light.color.r as f32 / 255.0,
                scene.environment_light.color.g as f32 / 255.0,
                scene.environment_light.color.b as f32 / 255.0,
            ],
            environment_rotation: scene.environment_light.rotation_radians,
            background_color: bg_clear_color,
        };

        sort_mesh_draws_3d(&mut mesh_draws_3d);

        PreparedScene {
            width: scene_config.output_width,
            height: scene_config.output_height,
            camera_uniform,
            camera_uniform_2d,
            sdf_primitives,
            surface_materials,
            mesh_vertices,
            mesh_indices,
            mesh_draws_3d,
            grids_3d,
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
    use crate::mobjects::mesh_3d::{SurfaceMaterial, TriangleMesh3D};
    use crate::mobjects::object_3d::{SdfPrimitive, Sphere3D};
    use crate::mobjects::{GridPlane, GridPlane3D, GridStyle3D, Rectangle};
    use crate::{Color, Scene, SceneConfig};

    fn prepare_transformed_triangle(
        transform: nalgebra::Matrix4<crate::GMFloat>,
    ) -> super::PreparedScene {
        let normal = Vector3::new(-1.0, -1.0, 1.0).normalize();
        let vertices = [[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]].map(|position| {
            crate::mobjects::mesh_3d::Vertex {
                position,
                normal: normal.into(),
                color: [1.0; 4],
                surface_coord: [0.0; 3],
            }
        });
        let mut scene = Scene::default();
        let id = scene.add(TriangleMesh3D::new(vertices.to_vec(), vec![0, 1, 2]));
        scene.world.get_mut(id).unwrap().set_transform(transform);
        ScenePreparer::default().prepare(&scene, &SceneConfig::default(), 1)
    }

    #[test]
    fn mesh_normals_remain_perpendicular_after_scaling_shear_and_reflection() {
        for x_scale in [2.0, -2.0] {
            let mut transform =
                nalgebra::Matrix4::new_nonuniform_scaling(&Vector3::new(x_scale, 3.0, 0.5));
            transform[(0, 1)] = 0.4;
            transform[(0, 3)] = 7.0;
            let prepared = prepare_transformed_triangle(transform);
            let positions: Vec<_> = prepared
                .mesh_vertices
                .iter()
                .map(|vertex| Vector3::from(vertex.position))
                .collect();
            let normal = Vector3::from(prepared.mesh_vertices[0].normal);
            assert!((normal.norm() - 1.0).abs() < 1e-6);
            assert!(normal.dot(&(positions[1] - positions[0])).abs() < 1e-6);
            assert!(normal.dot(&(positions[2] - positions[0])).abs() < 1e-6);
            // Reflections must preserve the inverse-transpose normal orientation.
            assert!(normal.dot(&transform.transform_vector(&Vector3::new(-1.0, -1.0, 1.0))) > 0.0);
        }
    }

    #[test]
    fn singular_mesh_transforms_keep_finite_normals() {
        for scale in [Vector3::new(2.0, 3.0, 0.0), Vector3::zeros()] {
            let prepared =
                prepare_transformed_triangle(nalgebra::Matrix4::new_nonuniform_scaling(&scale));
            assert_eq!(prepared.mesh_indices.len(), 3);
            for vertex in prepared.mesh_vertices {
                let normal = Vector3::from(vertex.normal);
                assert!(normal.iter().all(|value| value.is_finite()));
                assert!((normal - Vector3::z()).norm() < 1e-6);
            }
        }
    }

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
        scene.add(GridPlane3D::new(
            GridPlane::Xz,
            Point3::new(0.0, -1.0, 0.0),
            40.0,
            GridStyle3D::default(),
        ));

        let config = SceneConfig::default();
        let mut preparer = ScenePreparer::default();
        let first = preparer.prepare(&scene, &config, 1);

        assert_eq!(first.sdf_primitives.len(), 1);
        assert_eq!(first.camera_uniform.num_primitives, 1);
        assert_eq!(first.surface_materials.len(), 2);
        assert!(!first.mesh_vertices.is_empty());
        assert!(!first.mesh_indices.is_empty());
        assert_eq!(first.mesh_draws_3d.len(), 1);
        assert_eq!(first.grids_3d.len(), 1);
        assert_eq!(first.grids_3d[0].origin, Point3::new(0.0, -1.0, 0.0));
        assert_eq!(first.grids_3d[0].u_axis, Vector3::x());
        assert_eq!(first.grids_3d[0].v_axis, Vector3::z());
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
        assert_eq!(
            first.mesh_batches_2d[0].geometry.id(),
            second.mesh_batches_2d[0].geometry.id(),
        );
    }

    #[test]
    fn mesh_3d_world_transform_cache_tracks_geometry_id_and_transform() {
        let mut scene = Scene::default();
        let box_id = scene.add(TriangleMesh3D::box_mesh(
            Point3::new(0.0, 0.0, -3.0),
            Vector3::new(0.5, 0.5, 0.5),
            Color::white(),
        ));
        let config = SceneConfig::default();
        let mut preparer = ScenePreparer::default();

        let first = preparer.prepare(&scene, &config, 1);
        assert_eq!(preparer.mesh_cache_3d.len(), 1);
        let second = preparer.prepare(&scene, &config, 1);
        assert_eq!(preparer.mesh_cache_3d.len(), 1);
        assert_eq!(first.mesh_vertices, second.mesh_vertices);

        scene
            .world
            .get_mut(box_id)
            .unwrap()
            .move_by(Vector3::new(1.0, 0.0, 0.0));
        let moved = preparer.prepare(&scene, &config, 1);
        assert_eq!(preparer.mesh_cache_3d.len(), 1);
        assert_ne!(first.mesh_vertices, moved.mesh_vertices);

        scene.world.remove(box_id).unwrap();
        let _ = preparer.prepare(&scene, &config, 1);
        assert!(preparer.mesh_cache_3d.is_empty());
    }

    #[test]
    fn opaque_draws_are_sorted_front_to_back_and_transparent_back_to_front() {
        let draw = |view_depth, transparent| super::Mesh3DDraw {
            first_index: 0,
            index_count: 3,
            material_index: 0,
            transparent,
            view_depth,
        };
        let mut draws = vec![
            draw(5.0, false),
            draw(1.0, true),
            draw(1.0, false),
            draw(5.0, true),
            draw(3.0, false),
            draw(3.0, true),
        ];
        super::sort_mesh_draws_3d(&mut draws);
        assert_eq!(
            draws
                .iter()
                .map(|draw| (draw.is_transparent(), draw.view_depth))
                .collect::<Vec<_>>(),
            vec![
                (false, 1.0),
                (false, 3.0),
                (false, 5.0),
                (true, 5.0),
                (true, 3.0),
                (true, 1.0),
            ]
        );
    }
}
