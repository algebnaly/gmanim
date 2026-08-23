use crate::mobjects::object_3d::SdfPrimitive;

use super::mesh_2d::PreparedMesh2D;
use super::output::RenderOutputs;
use super::profiling::RendererStats;
use super::record::{RecordingPlan, RecordingPlanInput};
use super::scene::PreparedScene;

const COPY_BYTES_PER_ROW_ALIGNMENT: u32 = 256;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct GpuSdfPrimitive {
    pub(super) material_index: u32,
    pub(super) shape_type: u32,
    pub(super) padding: [u32; 2],
    pub(super) params: [f32; 12],
}

impl GpuSdfPrimitive {
    pub(super) fn encode(primitive: SdfPrimitive, material_index: u32) -> Self {
        let (shape_type, params) = match primitive {
            SdfPrimitive::Sphere { center, radius } => (
                0,
                [
                    center.x as f32,
                    center.y as f32,
                    center.z as f32,
                    radius as f32,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            ),
            SdfPrimitive::Capsule { start, end, radius } => (
                1,
                [
                    start.x as f32,
                    start.y as f32,
                    start.z as f32,
                    end.x as f32,
                    end.y as f32,
                    end.z as f32,
                    radius as f32,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            ),
            SdfPrimitive::Arrow {
                start,
                end,
                shaft_radius,
                head_radius,
                head_length,
            } => (
                2,
                [
                    start.x as f32,
                    start.y as f32,
                    start.z as f32,
                    end.x as f32,
                    end.y as f32,
                    end.z as f32,
                    shaft_radius as f32,
                    head_radius as f32,
                    head_length as f32,
                    0.0,
                    0.0,
                    0.0,
                ],
            ),
            SdfPrimitive::OrientedBox {
                center,
                half_extents,
                x_axis,
                y_axis,
            } => (
                3,
                [
                    center.x as f32,
                    center.y as f32,
                    center.z as f32,
                    half_extents.x as f32,
                    half_extents.y as f32,
                    half_extents.z as f32,
                    x_axis.x as f32,
                    x_axis.y as f32,
                    x_axis.z as f32,
                    y_axis.x as f32,
                    y_axis.y as f32,
                    y_axis.z as f32,
                ],
            ),
            SdfPrimitive::QuadraticBezier {
                start,
                control,
                end,
                radius,
            } => (
                4,
                [
                    start.x as f32,
                    start.y as f32,
                    start.z as f32,
                    control.x as f32,
                    control.y as f32,
                    control.z as f32,
                    end.x as f32,
                    end.y as f32,
                    end.z as f32,
                    radius as f32,
                    0.0,
                    0.0,
                ],
            ),
        };
        Self {
            material_index,
            shape_type,
            padding: [0; 2],
            params,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct FrameRequirements {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) rgba_row_bytes: u32,
    pub(super) padded_rgba_row_bytes: u32,
    pub(super) raster_gbuffer: bool,
    pub(super) overlay_hdr: bool,
}

impl FrameRequirements {
    pub(super) fn for_scene(scene: &PreparedScene) -> Self {
        let has_opaque_meshes = scene
            .mesh_draws_3d
            .iter()
            .any(|draw| !draw.is_transparent());
        let has_transparent_meshes = scene.mesh_draws_3d.iter().any(|draw| draw.is_transparent());
        Self::from_features(
            scene.width,
            scene.height,
            !scene.sdf_primitives.is_empty(),
            has_opaque_meshes,
            has_transparent_meshes,
            !scene.mesh_batches_2d.is_empty(),
        )
    }

    fn from_features(
        width: u32,
        height: u32,
        has_sdf: bool,
        has_opaque_meshes: bool,
        has_transparent_meshes: bool,
        has_mesh_2d: bool,
    ) -> Self {
        let rgba_row_bytes = width * 4;
        let padded_rgba_row_bytes =
            rgba_row_bytes.div_ceil(COPY_BYTES_PER_ROW_ALIGNMENT) * COPY_BYTES_PER_ROW_ALIGNMENT;
        Self {
            width,
            height,
            rgba_row_bytes,
            padded_rgba_row_bytes,
            raster_gbuffer: has_opaque_meshes,
            overlay_hdr: (has_sdf || has_opaque_meshes) && (has_transparent_meshes || has_mesh_2d),
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct FrameOptions {
    pub(super) ssaa_factor: u32,
    pub(super) analytic_aa_2d: bool,
    pub(super) bloom_enabled: bool,
    pub(super) gpu_profiling: bool,
    pub(super) outputs: RenderOutputs,
}

pub(super) struct PreparedFrame {
    pub(super) scene: PreparedScene,
    pub(super) sdf_primitives: Vec<GpuSdfPrimitive>,
    pub(super) mesh_2d: PreparedMesh2D,
    pub(super) plan: RecordingPlan,
    pub(super) outputs: RenderOutputs,
    pub(super) stats: RendererStats,
}

impl PreparedFrame {
    pub(super) fn new(
        scene: PreparedScene,
        requirements: FrameRequirements,
        mesh_2d: PreparedMesh2D,
        mesh_2d_arena_rebuilds: u32,
        options: FrameOptions,
    ) -> Self {
        let sdf_primitives = scene
            .sdf_primitives
            .iter()
            .map(|prepared| GpuSdfPrimitive::encode(prepared.primitive, prepared.material_index))
            .collect::<Vec<_>>();
        let has_opaque_meshes = scene
            .mesh_draws_3d
            .iter()
            .any(|draw| !draw.is_transparent());
        let has_transparent_meshes = scene.mesh_draws_3d.iter().any(|draw| draw.is_transparent());
        let plan = RecordingPlan::new(RecordingPlanInput {
            width: requirements.width,
            height: requirements.height,
            ssaa_factor: options.ssaa_factor,
            analytic_aa_2d: options.analytic_aa_2d,
            bloom_enabled: options.bloom_enabled,
            gpu_profiling: options.gpu_profiling,
            has_sdf: !sdf_primitives.is_empty(),
            has_mesh_indices: !scene.mesh_indices.is_empty(),
            has_prepared_mesh_2d: !mesh_2d.batches.is_empty(),
            all_2d_analytic: mesh_2d
                .instances
                .iter()
                .all(|instance| instance.aa_params[2] > 0.5),
            has_opaque_meshes,
            has_transparent_meshes,
            outputs: options.outputs,
            background_color: scene.background_color,
            camera_clip: (scene.camera_uniform.has_clip != 0).then_some([
                scene.camera_uniform.clip_x,
                scene.camera_uniform.clip_y,
                scene.camera_uniform.clip_w,
                scene.camera_uniform.clip_h,
            ]),
            camera_raster_scale: scene.camera_uniform.raster_scale as f32,
        });
        let stats = plan.stats(
            &scene.mesh_draws_3d,
            mesh_2d.batches.len() as u32,
            mesh_2d.instances.len() as u32,
            &mesh_2d.uploads,
            mesh_2d_arena_rebuilds,
            options.outputs,
        );
        Self {
            scene,
            sdf_primitives,
            mesh_2d,
            plan,
            outputs: options.outputs,
            stats,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::FrameRequirements;

    #[test]
    fn rgba_rows_are_aligned_for_buffer_copies() {
        let requirements = FrameRequirements::from_features(1921, 1080, false, false, false, false);
        assert_eq!(requirements.rgba_row_bytes, 7684);
        assert_eq!(requirements.padded_rgba_row_bytes, 7936);
    }

    #[test]
    fn target_features_follow_surface_composition() {
        let sdf_with_2d = FrameRequirements::from_features(320, 180, true, false, false, true);
        assert!(!sdf_with_2d.raster_gbuffer);
        assert!(sdf_with_2d.overlay_hdr);

        let opaque_only = FrameRequirements::from_features(320, 180, false, true, false, false);
        assert!(opaque_only.raster_gbuffer);
        assert!(!opaque_only.overlay_hdr);

        let transparent_only =
            FrameRequirements::from_features(320, 180, false, false, true, false);
        assert!(!transparent_only.raster_gbuffer);
        assert!(!transparent_only.overlay_hdr);
    }
}
