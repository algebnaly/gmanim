use crate::mobjects::object_3d::SdfPrimitive;

use super::CameraUniform;
use super::mesh_2d::PreparedMesh2D;
use super::output::RenderOutputs;
use super::profiling::RendererStats;
use super::record::{RecordingPlan, RecordingPlanInput};
use super::scene::{PreparedGrid3D, PreparedScene};

const COPY_BYTES_PER_ROW_ALIGNMENT: u32 = 256;
pub(super) const GRID_QUAD_VERTEX_COUNT: u32 = 6;

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
}

impl GpuGrid3D {
    fn prepare(grid: &PreparedGrid3D, camera: &CameraUniform) -> Self {
        Self {
            origin: [
                grid.origin.x,
                grid.origin.y,
                grid.origin.z,
                grid.half_extent,
            ],
            u_axis: [grid.u_axis.x, grid.u_axis.y, grid.u_axis.z, 0.0],
            v_axis: [grid.v_axis.x, grid.v_axis.y, grid.v_axis.z, 0.0],
            major_color: grid.style.major_color,
            minor_color: grid.style.minor_color,
            u_axis_color: grid.style.u_axis_color,
            v_axis_color: grid.style.v_axis_color,
            params: [
                grid.style.line_width_pixels * camera.raster_scale.max(1) as f32,
                grid.style.fade_radius,
                grid.style.cell_size,
                grid.style.subdivisions.max(1) as f32,
            ],
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct FrameRequirements {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) rgba_row_bytes: u32,
    pub(super) padded_rgba_row_bytes: u32,
    pub(super) raster_width: u32,
    pub(super) raster_height: u32,
    pub(super) multisampled_raster: bool,
    pub(super) depth: bool,
    pub(super) rgba_image: bool,
    pub(super) sdf: bool,
    pub(super) surface: bool,
    pub(super) raster_gbuffer: bool,
    pub(super) overlay_hdr: bool,
    pub(super) transparent: bool,
    pub(super) bloom: bool,
    pub(super) outputs: RenderOutputs,
}

impl FrameRequirements {
    pub(super) fn for_plan(plan: RecordingPlan, outputs: RenderOutputs) -> Self {
        let width = plan.width;
        let height = plan.height;
        let rgba_row_bytes = width * 4;
        let padded_rgba_row_bytes =
            rgba_row_bytes.div_ceil(COPY_BYTES_PER_ROW_ALIGNMENT) * COPY_BYTES_PER_ROW_ALIGNMENT;
        let has_content = plan.execution.runs_sdf() || plan.execution.runs_raster();
        Self {
            width,
            height,
            rgba_row_bytes,
            padded_rgba_row_bytes,
            raster_width: if has_content {
                width * plan.raster_scale
            } else {
                1
            },
            raster_height: if has_content {
                height * plan.raster_scale
            } else {
                1
            },
            multisampled_raster: plan.execution.runs_raster()
                && !plan.analytic_2d
                && (!plan.uses_deferred_raster || plan.has_surface_overlay),
            depth: plan.raster_uses_depth,
            rgba_image: !plan.fused_video_downsample,
            sdf: plan.execution.runs_sdf(),
            surface: plan.execution.runs_sdf() || plan.uses_deferred_raster,
            raster_gbuffer: plan.uses_deferred_raster,
            overlay_hdr: plan.has_surface_overlay,
            transparent: plan.has_transparent_meshes,
            bloom: has_content && plan.bloom_enabled,
            outputs,
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
            .map(|grid| GpuGrid3D::prepare(grid, &scene.camera_uniform))
            .collect::<Vec<_>>();
        let has_opaque_meshes = scene
            .mesh_draws_3d
            .iter()
            .any(|draw| !draw.is_transparent());
        let has_transparent_meshes = scene.mesh_draws_3d.iter().any(|draw| draw.is_transparent());
        let plan = RecordingPlan::new(RecordingPlanInput {
            width: scene.width,
            height: scene.height,
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
            grid_3d_draw_calls: grids_3d.len() as u32,
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
    use super::{CameraUniform, FrameRequirements, GpuGrid3D, PreparedGrid3D};
    use crate::mobjects::GridStyle3D;
    use bytemuck::Zeroable;
    use nalgebra::{Point3, Vector3};

    fn requirements_for_features(
        width: u32,
        height: u32,
        sdf: bool,
        opaque: bool,
        transparent: bool,
        mesh_2d: bool,
        grid: bool,
    ) -> FrameRequirements {
        let outputs = super::RenderOutputs::CPU_RGBA_ONLY;
        let plan = super::RecordingPlan::new(super::RecordingPlanInput {
            width,
            height,
            ssaa_factor: 2,
            analytic_aa_2d: true,
            bloom_enabled: false,
            gpu_profiling: false,
            has_sdf: sdf,
            has_mesh_indices: opaque || transparent,
            has_prepared_mesh_2d: mesh_2d,
            all_2d_analytic: true,
            has_opaque_meshes: opaque,
            has_transparent_meshes: transparent,
            grid_3d_draw_calls: u32::from(grid),
            outputs,
            background_color: [0.0; 4],
            camera_clip: None,
            camera_raster_scale: 2.0,
        });
        FrameRequirements::for_plan(plan, outputs)
    }

    #[test]
    fn grid_shader_abi_is_eight_vec4_values() {
        assert_eq!(std::mem::size_of::<GpuGrid3D>(), 8 * 16);
    }

    fn perspective_camera(pos: [f32; 3], target: [f32; 3], up: [f32; 3]) -> CameraUniform {
        let look =
            Vector3::new(target[0] - pos[0], target[1] - pos[1], target[2] - pos[2]).normalize();
        let mut camera = CameraUniform::zeroed();
        camera.pos = pos;
        camera.look_at = [look.x, look.y, look.z];
        camera.up = up;
        camera.fov = std::f32::consts::FRAC_PI_4;
        camera.width = 1920.0;
        camera.height = 1080.0;
        camera.raster_scale = 1;
        camera.proj_mat = crate::camera::Projection::perspective_wgpu(
            std::f32::consts::FRAC_PI_4,
            16.0 / 9.0,
            0.1,
            50.0,
        );
        camera
    }

    #[test]
    fn grid_gpu_payload_is_world_geometry() {
        let style = GridStyle3D {
            cell_size: 4.0,
            subdivisions: 5,
            ..GridStyle3D::default()
        };
        let grid = PreparedGrid3D {
            origin: Point3::origin(),
            u_axis: Vector3::x(),
            v_axis: Vector3::y(),
            half_extent: 50.0,
            style,
        };
        let looking_down = perspective_camera([1.0, -6.0, 5.0], [2.0, 2.0, 2.0], [0.0, 0.0, 1.0]);
        let after_orbit = perspective_camera([9.17, 0.58, 6.53], [2.0, 2.0, 2.0], [0.0, 0.0, 1.0]);
        let a = GpuGrid3D::prepare(&grid, &looking_down);
        let b = GpuGrid3D::prepare(&grid, &after_orbit);
        assert_eq!(a.origin, b.origin);
        assert_eq!(a.u_axis, b.u_axis);
        assert_eq!(a.v_axis, b.v_axis);
        assert_eq!(a.params[2], 4.0);
        assert_eq!(a.params[3], 5.0);
        assert_eq!(a.origin[3], 50.0);
    }

    #[test]
    fn rgba_rows_are_aligned_for_buffer_copies() {
        let requirements = requirements_for_features(1921, 1080, false, false, false, false, false);
        assert_eq!(requirements.rgba_row_bytes, 7684);
        assert_eq!(requirements.padded_rgba_row_bytes, 7936);
    }

    #[test]
    fn target_features_follow_surface_composition() {
        let sdf_with_2d = requirements_for_features(320, 180, true, false, false, true, false);
        assert!(!sdf_with_2d.raster_gbuffer);
        assert!(sdf_with_2d.overlay_hdr);

        let opaque_only = requirements_for_features(320, 180, false, true, false, false, false);
        assert!(opaque_only.raster_gbuffer);
        assert!(!opaque_only.overlay_hdr);

        let transparent_only =
            requirements_for_features(320, 180, false, false, true, false, false);
        assert!(!transparent_only.raster_gbuffer);
        assert!(!transparent_only.overlay_hdr);

        let grid_only = requirements_for_features(320, 180, false, false, false, false, true);
        assert!(!grid_only.raster_gbuffer);
        assert!(!grid_only.overlay_hdr);

        let sdf_with_grid = requirements_for_features(320, 180, true, false, false, false, true);
        assert!(!sdf_with_grid.raster_gbuffer);
        assert!(sdf_with_grid.overlay_hdr);
    }

    #[test]
    fn target_extents_follow_the_selected_aa_path() {
        let empty = requirements_for_features(320, 180, false, false, false, false, false);
        assert_eq!((empty.raster_width, empty.raster_height), (1, 1));
        let analytic = requirements_for_features(320, 180, false, false, false, true, false);
        assert_eq!((analytic.raster_width, analytic.raster_height), (320, 180));
        assert!(!analytic.sdf && !analytic.surface && !analytic.transparent && !analytic.bloom);
        let mixed = requirements_for_features(320, 180, true, false, false, true, false);
        assert_eq!((mixed.raster_width, mixed.raster_height), (640, 360));
        assert!(mixed.sdf && mixed.surface && mixed.overlay_hdr);
    }
}
