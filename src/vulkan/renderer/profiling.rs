use super::frame::{FrameExecutionPlan, GPU_TIMESTAMP_COUNT};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RendererStats {
    pub mesh_3d_opaque_draw_calls: u32,
    pub mesh_3d_transparent_draw_calls: u32,
    pub grid_3d_draw_calls: u32,
    pub mesh_2d_draw_calls: u32,
    pub mesh_2d_instances: u32,
    pub mesh_2d_geometry_uploads: u32,
    pub mesh_2d_vertex_bytes_uploaded: u64,
    pub mesh_2d_index_bytes_uploaded: u64,
    pub mesh_2d_arena_rebuilds: u32,
    pub mesh_2d_analytic_aa: u32,
    pub sdf_dispatches: u32,
    pub surface_lighting_dispatches: u32,
    pub raster_passes: u32,
    pub depth_attachment_raster_passes: u32,
    pub tone_map_dispatches: u32,
    pub bloom_dispatches: u32,
    pub downsample_dispatches: u32,
    pub fused_video_downsample_dispatches: u32,
    pub surface_resolve_dispatches: u32,
    pub surface_composite_dispatches: u32,
    pub output_conversion_dispatches: u32,
    pub rgba_readback_copies: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GpuPassTimings {
    pub frame_ms: f64,
    pub geometry_upload_ms: f64,
    pub sdf_ms: f64,
    pub raster_ms: f64,
    pub postprocess_ms: f64,
    pub output_ms: f64,
}

impl GpuPassTimings {
    pub(super) fn from_timestamps(
        timestamps: [u64; GPU_TIMESTAMP_COUNT as usize],
        timestamp_period_ns: f32,
        timestamp_valid_bits: u32,
        plan: FrameExecutionPlan,
        has_geometry_upload: bool,
        has_postprocess: bool,
        has_output: bool,
    ) -> Self {
        let elapsed_ms = |start: usize, end: usize| {
            timestamp_delta(timestamps[start], timestamps[end], timestamp_valid_bits) as f64
                * timestamp_period_ns as f64
                / 1_000_000.0
        };
        Self {
            frame_ms: elapsed_ms(0, 5),
            geometry_upload_ms: if has_geometry_upload {
                elapsed_ms(0, 1)
            } else {
                Default::default()
            },
            sdf_ms: if plan.runs_sdf() {
                elapsed_ms(1, 2)
            } else {
                Default::default()
            },
            raster_ms: if plan.runs_raster() {
                elapsed_ms(2, 3)
            } else {
                Default::default()
            },
            postprocess_ms: if has_postprocess {
                elapsed_ms(3, 4)
            } else {
                Default::default()
            },
            output_ms: if has_output {
                elapsed_ms(4, 5)
            } else {
                Default::default()
            },
        }
    }
}

pub(super) fn timestamp_delta(start: u64, end: u64, valid_bits: u32) -> u64 {
    let mask = if valid_bits >= 64 {
        u64::MAX
    } else {
        (1u64 << valid_bits) - 1
    };
    end.wrapping_sub(start) & mask
}
