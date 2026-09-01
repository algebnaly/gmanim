use crate::mobjects::object_3d::SdfPrimitive;
use nalgebra::{Matrix4, Point3, Vector3, Vector4};

use super::CameraUniform;
use super::mesh_2d::PreparedMesh2D;
use super::output::RenderOutputs;
use super::profiling::RendererStats;
use super::record::{RecordingPlan, RecordingPlanInput};
use super::scene::{PreparedGrid3D, PreparedScene};

const COPY_BYTES_PER_ROW_ALIGNMENT: u32 = 256;
pub(super) const GRID_LINE_COUNT: u32 = 151;
pub(super) const GRID_LOD_COUNT: u32 = 3;
pub(super) const GRID_AXIS_LINE_COUNT: u32 = 2;
pub(super) const GRID_INSTANCES_PER_GRID: u32 =
    GRID_LINE_COUNT * 2 * GRID_LOD_COUNT + GRID_AXIS_LINE_COUNT;
const GRID_LEVEL_SEARCH_COUNT: u32 = 8;

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
    pub(super) lod: [f32; 4],
}

impl GpuGrid3D {
    fn prepare(grid: &PreparedGrid3D, camera: &CameraUniform) -> Self {
        let camera_position = Point3::new(camera.pos[0], camera.pos[1], camera.pos[2]);
        let view_direction =
            Vector3::new(camera.look_at[0], camera.look_at[1], camera.look_at[2]).normalize();
        let plane_normal = grid.u_axis.cross(&grid.v_axis).normalize();
        let plane_denominator = plane_normal.dot(&view_direction);
        let focus_position = if plane_denominator.abs() > 1e-4 {
            let distance = plane_normal.dot(&(grid.origin - camera_position)) / plane_denominator;
            if distance > 0.0 {
                camera_position + view_direction * distance
            } else {
                camera_position
            }
        } else {
            camera_position
        };
        let camera_delta = focus_position - grid.origin;
        let camera_u = local_coordinate(camera_delta, grid.u_axis);
        let camera_v = local_coordinate(camera_delta, grid.v_axis);
        let focus_world = grid.origin + grid.u_axis * camera_u + grid.v_axis * camera_v;
        let view_projection = view_projection(camera);

        let subdivisions = grid.style.subdivisions.max(1) as f32;
        let mut spacing = grid.style.cell_size / subdivisions;
        if subdivisions > 1.0 {
            for _ in 0..GRID_LEVEL_SEARCH_COUNT {
                let pixel_spacing = grid_pixel_spacing(
                    focus_world,
                    grid.u_axis,
                    grid.v_axis,
                    spacing,
                    view_projection,
                    camera,
                );
                if pixel_spacing >= 2.0 {
                    break;
                }
                spacing *= subdivisions;
            }
        }

        let spacings = [
            spacing,
            spacing * subdivisions,
            spacing * subdivisions * subdivisions,
        ];
        let alphas = spacings.map(|lod_spacing| {
            smoothstep(
                2.0,
                8.0,
                grid_pixel_spacing(
                    focus_world,
                    grid.u_axis,
                    grid.v_axis,
                    lod_spacing,
                    view_projection,
                    camera,
                ),
            )
        });

        Self {
            origin: [
                grid.origin.x,
                grid.origin.y,
                grid.origin.z,
                grid.half_extent,
            ],
            u_axis: [grid.u_axis.x, grid.u_axis.y, grid.u_axis.z, camera_u],
            v_axis: [grid.v_axis.x, grid.v_axis.y, grid.v_axis.z, camera_v],
            major_color: grid.style.major_color,
            minor_color: grid.style.minor_color,
            u_axis_color: grid.style.u_axis_color,
            v_axis_color: grid.style.v_axis_color,
            params: [
                grid.style.line_width_pixels * camera.raster_scale.max(1) as f32,
                grid.style.fade_radius,
                alphas[0],
                alphas[1],
            ],
            lod: [spacings[0], spacings[1], spacings[2], alphas[2]],
        }
    }
}

fn local_coordinate(delta: Vector3<f32>, axis: Vector3<f32>) -> f32 {
    delta.dot(&axis) / axis.norm_squared().max(1e-8)
}

fn view_projection(camera: &CameraUniform) -> Matrix4<f32> {
    let position = Vector3::new(camera.pos[0], camera.pos[1], camera.pos[2]);
    let look = Vector3::new(camera.look_at[0], camera.look_at[1], camera.look_at[2]);
    let up = Vector3::new(camera.up[0], camera.up[1], camera.up[2]);
    let w = -look.normalize();
    let u = up.cross(&w).normalize();
    let v = w.cross(&u);
    let view = Matrix4::new(
        u.x,
        u.y,
        u.z,
        -u.dot(&position),
        v.x,
        v.y,
        v.z,
        -v.dot(&position),
        w.x,
        w.y,
        w.z,
        -w.dot(&position),
        0.0,
        0.0,
        0.0,
        1.0,
    );
    Matrix4::from_column_slice(&camera.proj_mat) * view
}

fn projected_spacing_pixels(
    world_position: Point3<f32>,
    offset: Vector3<f32>,
    view_projection: Matrix4<f32>,
    camera: &CameraUniform,
) -> f32 {
    let a = view_projection * world_position.to_homogeneous();
    let b = view_projection * (world_position + offset).to_homogeneous();
    if a.w <= 1e-4 || b.w <= 1e-4 {
        return 0.0;
    }
    let delta = Vector4::new(
        (b.x / b.w - a.x / a.w) * camera.width * 0.5,
        (b.y / b.w - a.y / a.w) * camera.height * 0.5,
        0.0,
        0.0,
    );
    delta.xy().norm()
}

fn grid_pixel_spacing(
    focus_world: Point3<f32>,
    u_axis: Vector3<f32>,
    v_axis: Vector3<f32>,
    spacing: f32,
    view_projection: Matrix4<f32>,
    camera: &CameraUniform,
) -> f32 {
    let u = projected_spacing_pixels(focus_world, u_axis * spacing, view_projection, camera);
    let v = projected_spacing_pixels(focus_world, v_axis * spacing, view_projection, camera);
    // A probe that lands behind the camera returns 0. Using min() would then
    // collapse every LOD alpha, including the coarse level that used to own
    // the world axes, so ignore invalid probes when the other axis is visible.
    match (u > 1e-4, v > 1e-4) {
        (true, true) => u.min(v),
        (true, false) => u,
        (false, true) => v,
        (false, false) => 0.0,
    }
}

fn smoothstep(edge0: f32, edge1: f32, value: f32) -> f32 {
    let t = ((value - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
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
            .map(|grid| GpuGrid3D::prepare(grid, &scene.camera_uniform))
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
    use super::{
        GRID_AXIS_LINE_COUNT, GRID_INSTANCES_PER_GRID, GRID_LINE_COUNT, GRID_LOD_COUNT,
        CameraUniform, FrameRequirements, GpuGrid3D, PreparedGrid3D,
    };
    use crate::mobjects::GridStyle3D;
    use bytemuck::Zeroable;
    use nalgebra::{Point3, Vector3};

    #[test]
    fn grid_shader_abi_is_nine_vec4_values() {
        assert_eq!(std::mem::size_of::<GpuGrid3D>(), 9 * 16);
    }

    #[test]
    fn grid_instances_include_dedicated_world_axes() {
        assert_eq!(
            GRID_INSTANCES_PER_GRID,
            GRID_LINE_COUNT * 2 * GRID_LOD_COUNT + GRID_AXIS_LINE_COUNT
        );
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
    fn coarse_grid_lod_stays_visible_from_oblique_z_up_view() {
        let camera = perspective_camera([9.17, 0.58, 6.53], [2.0, 2.0, 2.0], [0.0, 0.0, 1.0]);
        let grid = PreparedGrid3D {
            origin: Point3::origin(),
            u_axis: Vector3::x(),
            v_axis: Vector3::y(),
            half_extent: 50.0,
            style: GridStyle3D {
                cell_size: 4.0,
                subdivisions: 5,
                ..GridStyle3D::default()
            },
        };
        let gpu = GpuGrid3D::prepare(&grid, &camera);
        assert!(
            gpu.lod[3] > 0.5,
            "coarsest lod alpha collapsed to {}, world axes would vanish",
            gpu.lod[3]
        );
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
