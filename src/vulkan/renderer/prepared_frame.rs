use crate::mobjects::object_3d::SdfPrimitive;

use super::mesh_2d::PreparedMesh2D;
use super::output::RenderOutputs;
use super::profiling::RendererStats;
use super::record::{RecordingPlan, RecordingPlanInput};
use super::scene::{PreparedGrid3D, PreparedScene};

const COPY_BYTES_PER_ROW_ALIGNMENT: u32 = 256;
pub(super) const GRID_LINE_COUNT: u32 = 151;
pub(super) const GRID_LOD_COUNT: u32 = 3;
pub(super) const GRID_INSTANCES_PER_GRID: u32 = GRID_LINE_COUNT * 2 * GRID_LOD_COUNT;

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
                    center.x, center.y, center.z, radius, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ],
            ),
            SdfPrimitive::Capsule { start, end, radius } => (
                1,
                [
                    start.x, start.y, start.z, end.x, end.y, end.z, radius, 0.0, 0.0, 0.0, 0.0, 0.0,
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
                    start.x,
                    start.y,
                    start.z,
                    end.x,
                    end.y,
                    end.z,
                    shaft_radius,
                    head_radius,
                    head_length,
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
                    center.x,
                    center.y,
                    center.z,
                    half_extents.x,
                    half_extents.y,
                    half_extents.z,
                    x_axis.x,
                    x_axis.y,
                    x_axis.z,
                    y_axis.x,
                    y_axis.y,
                    y_axis.z,
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
                    start.x, start.y, start.z, control.x, control.y, control.z, end.x, end.y,
                    end.z, radius, 0.0, 0.0,
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

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct GpuGrid3D {
    pub(super) origin: [f32; 4],
    pub(super) u_axis: [f32; 4],
    pub(super) v_axis: [f32; 4],
    pub(super) major_color: [f32; 4],
    pub(super) minor_color: [f32; 4],
    pub(super) u_axis_color: [f32; 4],
    pub(super) v_axis_color: [f32; 4],
    pub(super) params: [f32; 4],
    pub(super) extent: [f32; 4],
}

impl From<&PreparedGrid3D> for GpuGrid3D {
    fn from(grid: &PreparedGrid3D) -> Self {
        Self {
            origin: [grid.origin.x, grid.origin.y, grid.origin.z, 1.0],
            u_axis: [grid.u_axis.x, grid.u_axis.y, grid.u_axis.z, 0.0],
            v_axis: [grid.v_axis.x, grid.v_axis.y, grid.v_axis.z, 0.0],
            major_color: grid.style.major_color,
            minor_color: grid.style.minor_color,
            u_axis_color: grid.style.u_axis_color,
            v_axis_color: grid.style.v_axis_color,
            params: [
                grid.style.cell_size,
                grid.style.subdivisions as f32,
                grid.style.line_width_pixels,
                grid.style.fade_radius,
            ],
            extent: [grid.half_extent, 0.0, 0.0, 0.0],
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
            !scene.grids_3d.is_empty(),
        )
    }

    fn from_features(
        width: u32,
        height: u32,
        has_sdf: bool,
        has_opaque_meshes: bool,
        has_transparent_meshes: bool,
        has_mesh_2d: bool,
        has_grid_3d: bool,
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
            overlay_hdr: (has_sdf || has_opaque_meshes)
                && (has_transparent_meshes || has_mesh_2d || has_grid_3d),
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
    pub(super) grids_3d: Vec<GpuGrid3D>,
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
        let grids_3d = scene
            .grids_3d
            .iter()
            .map(GpuGrid3D::from)
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
            has_grid_3d: !grids_3d.is_empty(),
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
            grids_3d,
            mesh_2d,
            plan,
            outputs: options.outputs,
            stats,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{FrameRequirements, GpuGrid3D};

    #[test]
    fn grid_shader_abi_is_nine_vec4_values() {
        assert_eq!(std::mem::size_of::<GpuGrid3D>(), 9 * 16);
    }

    #[test]
    fn rgba_rows_are_aligned_for_buffer_copies() {
        let requirements =
            FrameRequirements::from_features(1921, 1080, false, false, false, false, false);
        assert_eq!(requirements.rgba_row_bytes, 7684);
        assert_eq!(requirements.padded_rgba_row_bytes, 7936);
    }

    #[test]
    fn target_features_follow_surface_composition() {
        let sdf_with_2d =
            FrameRequirements::from_features(320, 180, true, false, false, true, false);
        assert!(!sdf_with_2d.raster_gbuffer);
        assert!(sdf_with_2d.overlay_hdr);

        let opaque_only =
            FrameRequirements::from_features(320, 180, false, true, false, false, false);
        assert!(opaque_only.raster_gbuffer);
        assert!(!opaque_only.overlay_hdr);

        let transparent_only =
            FrameRequirements::from_features(320, 180, false, false, true, false, false);
        assert!(!transparent_only.raster_gbuffer);
        assert!(!transparent_only.overlay_hdr);

        let grid_only =
            FrameRequirements::from_features(320, 180, false, false, false, false, true);
        assert!(!grid_only.raster_gbuffer);
        assert!(!grid_only.overlay_hdr);

        let sdf_with_grid =
            FrameRequirements::from_features(320, 180, true, false, false, false, true);
        assert!(!sdf_with_grid.raster_gbuffer);
        assert!(sdf_with_grid.overlay_hdr);
    }
}
