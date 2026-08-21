use std::collections::HashMap;
use std::sync::Arc;

use crate::mobjects::Rectangle;
use crate::mobjects::mesh_2d::{GeometryFingerprint, MeshGeometry2D, Vertex2D};

use super::align_up;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct Instance2D {
    pub(super) model_0: [f32; 4],
    pub(super) model_1: [f32; 4],
    pub(super) model_2: [f32; 4],
    pub(super) model_3: [f32; 4],
    pub(super) color: [f32; 4],
    /// [half_extent_x, half_extent_y, analytic_aa_enabled, unused].
    /// When `z > 0.5` the fragment shader derives edge coverage from the
    /// interpolated local coordinates instead of relying on MSAA.
    pub(super) aa_params: [f32; 4],
}

impl Instance2D {
    pub(super) fn new(
        transform: nalgebra::Matrix4<crate::GMFloat>,
        color: [f32; 4],
        aa_params: [f32; 4],
    ) -> Self {
        Self {
            model_0: [
                transform[(0, 0)] as f32,
                transform[(1, 0)] as f32,
                transform[(2, 0)] as f32,
                transform[(3, 0)] as f32,
            ],
            model_1: [
                transform[(0, 1)] as f32,
                transform[(1, 1)] as f32,
                transform[(2, 1)] as f32,
                transform[(3, 1)] as f32,
            ],
            model_2: [
                transform[(0, 2)] as f32,
                transform[(1, 2)] as f32,
                transform[(2, 2)] as f32,
                transform[(3, 2)] as f32,
            ],
            model_3: [
                transform[(0, 3)] as f32,
                transform[(1, 3)] as f32,
                transform[(2, 3)] as f32,
                transform[(3, 3)] as f32,
            ],
            color,
            aa_params,
        }
    }
}

pub(super) struct Mesh2DSubmission {
    pub(super) geometry: Arc<MeshGeometry2D>,
    pub(super) instance: Instance2D,
    pub(super) dynamic: bool,
}

pub(super) struct Mesh2DBatch {
    pub(super) geometry: Arc<MeshGeometry2D>,
    pub(super) instances: Vec<Instance2D>,
    pub(super) dynamic: bool,
}

pub(super) struct CachedRectangle2D {
    pub(super) geometry_revision: u64,
    pub(super) source: Rectangle,
    pub(super) geometry: Arc<MeshGeometry2D>,
}

#[derive(Clone)]
pub(super) struct CachedMesh2D {
    pub(super) geometry: Arc<MeshGeometry2D>,
    pub(super) vertex_offset: u64,
    pub(super) index_offset: u64,
    pub(super) index_count: u32,
}

pub(super) struct GeometryUpload2D {
    pub(super) geometry: Arc<MeshGeometry2D>,
    pub(super) staging_vertex_offset: u64,
    pub(super) staging_index_offset: u64,
    pub(super) device_vertex_offset: u64,
    pub(super) device_index_offset: u64,
}

pub(super) struct PreparedMesh2DBatch {
    pub(super) first_index: u32,
    pub(super) vertex_offset: i32,
    pub(super) index_count: u32,
    pub(super) first_instance: u32,
    pub(super) instance_count: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum PrepareMesh2DError {
    StaticArenaExhausted,
    FrameDynamicArenaExhausted,
    FrameStagingArenaExhausted,
    FrameInstanceArenaExhausted,
}

#[derive(Clone, Copy)]
pub(super) struct Mesh2DFrameArenas {
    pub(super) dynamic_vertex_base: u64,
    pub(super) dynamic_index_base: u64,
    pub(super) dynamic_vertex_capacity: u64,
    pub(super) dynamic_index_capacity: u64,
    pub(super) staging_vertex_capacity: u64,
    pub(super) staging_index_capacity: u64,
    pub(super) instance_capacity: u64,
}

pub(super) struct PreparedMesh2D {
    pub(super) batches: Vec<PreparedMesh2DBatch>,
    pub(super) uploads: Vec<GeometryUpload2D>,
    pub(super) instances: Vec<Instance2D>,
}

pub(super) struct Mesh2DUploadPlanner {
    cache: HashMap<GeometryFingerprint, Vec<CachedMesh2D>>,
    static_vertex_capacity: u64,
    static_index_capacity: u64,
    static_vertex_used: u64,
    static_index_used: u64,
}

impl Mesh2DUploadPlanner {
    pub(super) fn new(static_vertex_capacity: u64, static_index_capacity: u64) -> Self {
        Self {
            cache: HashMap::new(),
            static_vertex_capacity,
            static_index_capacity,
            static_vertex_used: 0,
            static_index_used: 0,
        }
    }

    pub(super) fn prepare(
        &mut self,
        arenas: Mesh2DFrameArenas,
        batches: &[Mesh2DBatch],
    ) -> Result<PreparedMesh2D, PrepareMesh2DError> {
        let (batches, uploads, instances) = prepare_mesh_2d_batches(
            &mut self.cache,
            &mut self.static_vertex_used,
            &mut self.static_index_used,
            self.static_vertex_capacity,
            self.static_index_capacity,
            arenas,
            batches,
        )?;
        Ok(PreparedMesh2D {
            batches,
            uploads,
            instances,
        })
    }

    pub(super) fn frame_arenas(
        &self,
        frame_index: u64,
        vertex_capacity: u64,
        index_capacity: u64,
        instance_capacity: u64,
    ) -> Mesh2DFrameArenas {
        Mesh2DFrameArenas {
            dynamic_vertex_base: self.static_vertex_capacity + frame_index * vertex_capacity,
            dynamic_index_base: self.static_index_capacity + frame_index * index_capacity,
            dynamic_vertex_capacity: vertex_capacity,
            dynamic_index_capacity: index_capacity,
            staging_vertex_capacity: vertex_capacity,
            staging_index_capacity: index_capacity,
            instance_capacity,
        }
    }

    pub(super) fn reset_static_arena(&mut self) {
        self.cache.clear();
        self.static_vertex_used = 0;
        self.static_index_used = 0;
    }
}

/// Returns `[half_extent_x, half_extent_y, enabled, 0]` when the rectangle
/// qualifies for analytic edge AA. Other geometry keeps the multisampled
/// supersampled raster path.
pub(super) fn rectangle_analytic_aa_params(rectangle: &Rectangle) -> [f32; 4] {
    use crate::GMFloat;
    let degenerate = |value: GMFloat| value.abs() <= GMFloat::EPSILON * 4.0;
    if !rectangle.draw_config.fill || rectangle.draw_config.stoke_width > 0.0 {
        return [0.0; 4];
    }
    let edge_x = rectangle.p1 - rectangle.p0;
    let edge_y = rectangle.p3 - rectangle.p0;
    let edge_x_len = edge_x.norm();
    let edge_y_len = edge_y.norm();
    if degenerate(edge_x_len) || degenerate(edge_y_len) {
        return [0.0; 4];
    }
    let perpendicular = edge_x.dot(&edge_y).abs() <= edge_x_len * edge_y_len * 1e-4;
    let closure = (rectangle.p2 - (rectangle.p1 + edge_y)).norm();
    let closes = closure <= (edge_x_len + edge_y_len) * 1e-4;
    if !perpendicular || !closes {
        return [0.0; 4];
    }
    let aa_mode = if rectangle.aa_mode >= 2.0 {
        rectangle.aa_mode
    } else {
        1.0
    };
    [
        (edge_x_len / 2.0) as f32,
        (edge_y_len / 2.0) as f32,
        aa_mode,
        0.0,
    ]
}

pub(super) fn build_ordered_mesh_2d_batches(
    submissions: Vec<Mesh2DSubmission>,
) -> Vec<Mesh2DBatch> {
    let mut batches: Vec<Mesh2DBatch> = Vec::new();
    for submission in submissions {
        if let Some(last) = batches.last_mut() {
            if last.dynamic == submission.dynamic
                && last.geometry.same_geometry(&submission.geometry)
            {
                last.instances.push(submission.instance);
                continue;
            }
        }
        batches.push(Mesh2DBatch {
            geometry: submission.geometry,
            instances: vec![submission.instance],
            dynamic: submission.dynamic,
        });
    }
    batches
}

fn prepare_mesh_2d_batches(
    mesh_cache: &mut HashMap<GeometryFingerprint, Vec<CachedMesh2D>>,
    static_vertex_used: &mut u64,
    static_index_used: &mut u64,
    static_vertex_capacity: u64,
    static_index_capacity: u64,
    arenas: Mesh2DFrameArenas,
    batches: &[Mesh2DBatch],
) -> Result<
    (
        Vec<PreparedMesh2DBatch>,
        Vec<GeometryUpload2D>,
        Vec<Instance2D>,
    ),
    PrepareMesh2DError,
> {
    let mut prepared = Vec::with_capacity(batches.len());
    let mut uploads = Vec::new();
    let mut instances = Vec::new();
    let mut staging_vertex_used = 0u64;
    let mut staging_index_used = 0u64;
    let mut dynamic_vertex_used = 0u64;
    let mut dynamic_index_used = 0u64;

    for batch in batches {
        if batch.geometry.indices().is_empty() || batch.instances.is_empty() {
            continue;
        }
        let cached = if batch.dynamic {
            let vertex_size = std::mem::size_of_val(batch.geometry.vertices()) as u64;
            let index_size = std::mem::size_of_val(batch.geometry.indices()) as u64;
            let device_vertex_offset =
                arenas.dynamic_vertex_base + align_up(dynamic_vertex_used, 4);
            let device_index_offset = arenas.dynamic_index_base + align_up(dynamic_index_used, 4);
            let staging_vertex_offset = align_up(staging_vertex_used, 4);
            let staging_index_offset = align_up(staging_index_used, 4);

            if device_vertex_offset + vertex_size
                > arenas.dynamic_vertex_base + arenas.dynamic_vertex_capacity
                || device_index_offset + index_size
                    > arenas.dynamic_index_base + arenas.dynamic_index_capacity
            {
                return Err(PrepareMesh2DError::FrameDynamicArenaExhausted);
            }
            if staging_vertex_offset + vertex_size > arenas.staging_vertex_capacity
                || staging_index_offset + index_size > arenas.staging_index_capacity
            {
                return Err(PrepareMesh2DError::FrameStagingArenaExhausted);
            }

            uploads.push(GeometryUpload2D {
                geometry: batch.geometry.clone(),
                staging_vertex_offset,
                staging_index_offset,
                device_vertex_offset,
                device_index_offset,
            });
            dynamic_vertex_used = device_vertex_offset + vertex_size - arenas.dynamic_vertex_base;
            dynamic_index_used = device_index_offset + index_size - arenas.dynamic_index_base;
            staging_vertex_used = staging_vertex_offset + vertex_size;
            staging_index_used = staging_index_offset + index_size;
            CachedMesh2D {
                geometry: batch.geometry.clone(),
                vertex_offset: device_vertex_offset,
                index_offset: device_index_offset,
                index_count: batch.geometry.indices().len() as u32,
            }
        } else {
            let fingerprint = batch.geometry.fingerprint();
            let cached = mesh_cache.get(&fingerprint).and_then(|entries| {
                entries
                    .iter()
                    .find(|entry| entry.geometry.same_geometry(&batch.geometry))
                    .cloned()
            });
            match cached {
                Some(cached) => cached,
                None => {
                    let vertex_size = std::mem::size_of_val(batch.geometry.vertices()) as u64;
                    let index_size = std::mem::size_of_val(batch.geometry.indices()) as u64;
                    let device_vertex_offset = align_up(*static_vertex_used, 4);
                    let device_index_offset = align_up(*static_index_used, 4);
                    let staging_vertex_offset = align_up(staging_vertex_used, 4);
                    let staging_index_offset = align_up(staging_index_used, 4);

                    if device_vertex_offset + vertex_size > static_vertex_capacity
                        || device_index_offset + index_size > static_index_capacity
                    {
                        return Err(PrepareMesh2DError::StaticArenaExhausted);
                    }
                    if staging_vertex_offset + vertex_size > arenas.staging_vertex_capacity
                        || staging_index_offset + index_size > arenas.staging_index_capacity
                    {
                        return Err(PrepareMesh2DError::FrameStagingArenaExhausted);
                    }

                    let cached = CachedMesh2D {
                        geometry: batch.geometry.clone(),
                        vertex_offset: device_vertex_offset,
                        index_offset: device_index_offset,
                        index_count: batch.geometry.indices().len() as u32,
                    };
                    mesh_cache
                        .entry(fingerprint)
                        .or_default()
                        .push(cached.clone());
                    uploads.push(GeometryUpload2D {
                        geometry: batch.geometry.clone(),
                        staging_vertex_offset,
                        staging_index_offset,
                        device_vertex_offset,
                        device_index_offset,
                    });
                    *static_vertex_used = device_vertex_offset + vertex_size;
                    *static_index_used = device_index_offset + index_size;
                    staging_vertex_used = staging_vertex_offset + vertex_size;
                    staging_index_used = staging_index_offset + index_size;
                    cached
                }
            }
        };

        let first_instance = instances.len() as u32;
        instances.extend_from_slice(&batch.instances);
        prepared.push(PreparedMesh2DBatch {
            first_index: (cached.index_offset / std::mem::size_of::<u32>() as u64) as u32,
            vertex_offset: (cached.vertex_offset / std::mem::size_of::<Vertex2D>() as u64) as i32,
            index_count: cached.index_count,
            first_instance,
            instance_count: batch.instances.len() as u32,
        });
    }

    if std::mem::size_of_val(instances.as_slice()) as u64 > arenas.instance_capacity {
        return Err(PrepareMesh2DError::FrameInstanceArenaExhausted);
    }
    Ok((prepared, uploads, instances))
}

#[cfg(test)]
mod tests {
    use nalgebra::Matrix4;

    use super::*;

    fn triangle(offset: f32) -> Arc<MeshGeometry2D> {
        Arc::new(MeshGeometry2D::new(
            vec![
                Vertex2D {
                    position: [offset, 0.0],
                    local: [offset, 0.0],
                },
                Vertex2D {
                    position: [offset + 1.0, 0.0],
                    local: [offset + 1.0, 0.0],
                },
                Vertex2D {
                    position: [offset, 1.0],
                    local: [offset, 1.0],
                },
            ],
            vec![0, 1, 2],
        ))
    }

    fn instance(color: [f32; 4]) -> Instance2D {
        Instance2D::new(Matrix4::identity(), color, [0.0; 4])
    }

    #[test]
    fn ordered_batching_only_merges_consecutive_equal_geometry() {
        let first = triangle(0.0);
        let equal_to_first = triangle(0.0);
        let second = triangle(2.0);
        let batches = build_ordered_mesh_2d_batches(vec![
            Mesh2DSubmission {
                geometry: first.clone(),
                instance: instance([1.0, 0.0, 0.0, 1.0]),
                dynamic: false,
            },
            Mesh2DSubmission {
                geometry: equal_to_first,
                instance: instance([0.0, 1.0, 0.0, 1.0]),
                dynamic: false,
            },
            Mesh2DSubmission {
                geometry: second,
                instance: instance([0.0, 0.0, 1.0, 1.0]),
                dynamic: false,
            },
            Mesh2DSubmission {
                geometry: first,
                instance: instance([1.0; 4]),
                dynamic: false,
            },
        ]);

        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0].instances.len(), 2);
        assert_eq!(batches[1].instances.len(), 1);
        assert_eq!(batches[2].instances.len(), 1);
    }

    #[test]
    fn persistent_geometry_is_only_uploaded_on_cache_miss() {
        let geometry = triangle(0.0);
        let batches = build_ordered_mesh_2d_batches(vec![
            Mesh2DSubmission {
                geometry: geometry.clone(),
                instance: instance([1.0; 4]),
                dynamic: false,
            },
            Mesh2DSubmission {
                geometry,
                instance: instance([0.5; 4]),
                dynamic: false,
            },
        ]);
        let mut planner = Mesh2DUploadPlanner::new(4096, 4096);
        let arenas = planner.frame_arenas(0, 4096, 4096, 4096);
        let first = planner.prepare(arenas, &batches).unwrap();
        let second = planner.prepare(arenas, &batches).unwrap();

        assert_eq!(first.batches.len(), 1);
        assert_eq!(first.instances.len(), 2);
        assert_eq!(first.uploads.len(), 1);
        assert_eq!(second.batches.len(), 1);
        assert_eq!(second.instances.len(), 2);
        assert!(second.uploads.is_empty());
    }

    #[test]
    fn dynamic_geometry_uses_its_frame_arena() {
        let batches = build_ordered_mesh_2d_batches(vec![Mesh2DSubmission {
            geometry: triangle(0.0),
            instance: instance([1.0; 4]),
            dynamic: true,
        }]);
        let mut planner = Mesh2DUploadPlanner::new(4096, 4096);

        for (frame_index, dynamic_base) in [(0, 4096), (1, 8192)] {
            let arenas = planner.frame_arenas(frame_index, 4096, 4096, 4096);
            let prepared = planner.prepare(arenas, &batches).unwrap();
            assert_eq!(prepared.uploads.len(), 1);
            assert_eq!(prepared.uploads[0].device_vertex_offset, dynamic_base);
        }

        assert!(planner.cache.is_empty());
        assert_eq!(planner.static_vertex_used, 0);
        assert_eq!(planner.static_index_used, 0);
    }

    #[test]
    fn persistent_arena_exhaustion_requests_generation_rebuild() {
        let batches = build_ordered_mesh_2d_batches(vec![Mesh2DSubmission {
            geometry: triangle(0.0),
            instance: instance([1.0; 4]),
            dynamic: false,
        }]);
        let mut planner = Mesh2DUploadPlanner::new(1, 1);
        let arenas = planner.frame_arenas(0, 4096, 4096, 4096);

        assert!(matches!(
            planner.prepare(arenas, &batches),
            Err(PrepareMesh2DError::StaticArenaExhausted)
        ));
    }
}
