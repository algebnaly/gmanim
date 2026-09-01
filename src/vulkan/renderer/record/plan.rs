use super::super::Mesh3DDraw;
use super::super::frame::FrameExecutionPlan;
use super::super::mesh_2d::GeometryUpload2D;
use super::super::output::RenderOutputs;
use super::super::profiling::RendererStats;

#[derive(Clone, Copy, Debug)]
pub(in crate::vulkan::renderer) struct RecordingPlanInput {
    pub(in crate::vulkan::renderer) width: u32,
    pub(in crate::vulkan::renderer) height: u32,
    pub(in crate::vulkan::renderer) ssaa_factor: u32,
    pub(in crate::vulkan::renderer) analytic_aa_2d: bool,
    pub(in crate::vulkan::renderer) bloom_enabled: bool,
    pub(in crate::vulkan::renderer) gpu_profiling: bool,
    pub(in crate::vulkan::renderer) has_sdf: bool,
    pub(in crate::vulkan::renderer) has_mesh_indices: bool,
    pub(in crate::vulkan::renderer) has_prepared_mesh_2d: bool,
    pub(in crate::vulkan::renderer) all_2d_analytic: bool,
    pub(in crate::vulkan::renderer) has_opaque_meshes: bool,
    pub(in crate::vulkan::renderer) has_transparent_meshes: bool,
    pub(in crate::vulkan::renderer) grid_3d_draw_calls: u32,
    pub(in crate::vulkan::renderer) outputs: RenderOutputs,
    pub(in crate::vulkan::renderer) background_color: [f32; 4],
    pub(in crate::vulkan::renderer) camera_clip: Option<[f32; 4]>,
    pub(in crate::vulkan::renderer) camera_raster_scale: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::vulkan::renderer) struct RecordingPlan {
    pub(in crate::vulkan::renderer) execution: FrameExecutionPlan,
    pub(in crate::vulkan::renderer) width: u32,
    pub(in crate::vulkan::renderer) height: u32,
    pub(in crate::vulkan::renderer) ssaa_factor: u32,
    pub(in crate::vulkan::renderer) raster_scale: u32,
    pub(in crate::vulkan::renderer) analytic_2d: bool,
    pub(in crate::vulkan::renderer) fused_video_downsample: bool,
    pub(in crate::vulkan::renderer) runs_postprocess: bool,
    pub(in crate::vulkan::renderer) bloom_enabled: bool,
    pub(in crate::vulkan::renderer) gpu_profiling: bool,
    pub(in crate::vulkan::renderer) raster_uses_depth: bool,
    pub(in crate::vulkan::renderer) has_transparent_meshes: bool,
    pub(in crate::vulkan::renderer) has_grid_3d: bool,
    grid_3d_draw_calls: u32,
    pub(in crate::vulkan::renderer) uses_deferred_raster: bool,
    pub(in crate::vulkan::renderer) has_surface_overlay: bool,
    pub(in crate::vulkan::renderer) background_color: [f32; 4],
    pub(in crate::vulkan::renderer) camera_clip: Option<[f32; 4]>,
    pub(in crate::vulkan::renderer) camera_raster_scale: f32,
}

impl RecordingPlan {
    pub(in crate::vulkan::renderer) fn new(input: RecordingPlanInput) -> Self {
        let has_grid_3d = input.grid_3d_draw_calls != 0;
        // Analytic AA applies only when every rastered 2D instance is a
        // filled rectangle: the frame then renders at output resolution with
        // one sample, and the tone-map downsample factor becomes one.
        // Bloom is excluded for now because its extract pass still derives
        // its sampling grid from the resolved image dimensions.
        let raster_2d_only =
            !input.has_sdf && !input.has_mesh_indices && !has_grid_3d && input.has_prepared_mesh_2d;
        let analytic_2d =
            raster_2d_only && input.analytic_aa_2d && !input.bloom_enabled && input.all_2d_analytic;
        let execution = if analytic_2d {
            FrameExecutionPlan::RasterToneMap
        } else {
            FrameExecutionPlan::build(
                input.has_sdf,
                input.has_mesh_indices || input.has_prepared_mesh_2d || has_grid_3d,
                input.ssaa_factor,
            )
        };
        let raster_scale = if analytic_2d { 1 } else { input.ssaa_factor };
        let fused_video_downsample = execution == FrameExecutionPlan::RasterDownsample
            && input.ssaa_factor == 2
            && !input.bloom_enabled
            && input.outputs.vulkan_video
            && !input.outputs.cpu_nv12
            && !input.outputs.cpu_yuv444p
            && !input.outputs.cpu_rgba;
        let uses_deferred_raster = input.has_opaque_meshes;
        Self {
            execution,
            width: input.width,
            height: input.height,
            ssaa_factor: input.ssaa_factor,
            raster_scale,
            analytic_2d,
            fused_video_downsample,
            runs_postprocess: execution != FrameExecutionPlan::Empty && !fused_video_downsample,
            bloom_enabled: input.bloom_enabled,
            gpu_profiling: input.gpu_profiling,
            raster_uses_depth: input.has_mesh_indices || has_grid_3d,
            has_transparent_meshes: input.has_transparent_meshes,
            has_grid_3d,
            grid_3d_draw_calls: input.grid_3d_draw_calls,
            uses_deferred_raster,
            has_surface_overlay: (execution.runs_sdf() || uses_deferred_raster)
                && (input.has_transparent_meshes || input.has_prepared_mesh_2d || has_grid_3d),
            background_color: input.background_color,
            camera_clip: input.camera_clip,
            camera_raster_scale: input.camera_raster_scale,
        }
    }

    pub(in crate::vulkan::renderer) fn stats(
        self,
        mesh_draws_3d: &[Mesh3DDraw],
        mesh_2d_draw_calls: u32,
        mesh_2d_instances: u32,
        geometry_uploads_2d: &[GeometryUpload2D],
        mesh_2d_arena_rebuilds: u32,
        outputs: RenderOutputs,
    ) -> RendererStats {
        let surface = self.execution.runs_sdf() || self.uses_deferred_raster;
        RendererStats {
            mesh_3d_opaque_draw_calls: mesh_draws_3d
                .iter()
                .filter(|draw| !draw.is_transparent())
                .count() as u32,
            mesh_3d_transparent_draw_calls: mesh_draws_3d
                .iter()
                .filter(|draw| draw.is_transparent())
                .count() as u32
                * 2,
            grid_3d_draw_calls: self.grid_3d_draw_calls,
            mesh_2d_draw_calls,
            mesh_2d_instances,
            mesh_2d_geometry_uploads: geometry_uploads_2d.len() as u32,
            mesh_2d_vertex_bytes_uploaded: geometry_uploads_2d
                .iter()
                .map(|upload| std::mem::size_of_val(upload.geometry.vertices()) as u64)
                .sum(),
            mesh_2d_index_bytes_uploaded: geometry_uploads_2d
                .iter()
                .map(|upload| std::mem::size_of_val(upload.geometry.indices()) as u64)
                .sum(),
            mesh_2d_arena_rebuilds,
            mesh_2d_analytic_aa: self.analytic_2d as u32,
            sdf_dispatches: self.execution.runs_sdf() as u32,
            surface_lighting_dispatches: surface as u32,
            raster_passes: if !self.execution.runs_raster() {
                0
            } else if self.uses_deferred_raster {
                1 + self.has_surface_overlay as u32 + self.has_transparent_meshes as u32
            } else {
                1 + self.has_transparent_meshes as u32 * 2
            },
            depth_attachment_raster_passes: if !self.raster_uses_depth {
                0
            } else if self.uses_deferred_raster {
                1 + self.has_surface_overlay as u32
            } else {
                1 + self.has_transparent_meshes as u32
            },
            tone_map_dispatches: self.runs_postprocess as u32,
            bloom_dispatches: if self.bloom_enabled && self.execution != FrameExecutionPlan::Empty {
                3
            } else {
                0
            },
            downsample_dispatches: (self.execution == FrameExecutionPlan::RasterDownsample
                && !self.fused_video_downsample) as u32,
            fused_video_downsample_dispatches: self.fused_video_downsample as u32,
            surface_resolve_dispatches: surface as u32,
            surface_composite_dispatches: surface as u32,
            output_conversion_dispatches: outputs.cpu_nv12 as u32
                + outputs.cpu_yuv444p as u32
                + outputs.vulkan_video as u32,
            rgba_readback_copies: outputs.cpu_rgba as u32,
        }
    }

    pub(in crate::vulkan::renderer) fn raster_extent(self) -> ash::vk::Extent2D {
        ash::vk::Extent2D {
            width: self.width * self.raster_scale,
            height: self.height * self.raster_scale,
        }
    }

    pub(in crate::vulkan::renderer) fn ssaa_extent(self) -> ash::vk::Extent2D {
        ash::vk::Extent2D {
            width: self.width * self.ssaa_factor,
            height: self.height * self.ssaa_factor,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vulkan::renderer::output::RenderOutputs;

    fn input() -> RecordingPlanInput {
        RecordingPlanInput {
            width: 320,
            height: 180,
            ssaa_factor: 2,
            analytic_aa_2d: true,
            bloom_enabled: false,
            gpu_profiling: false,
            has_sdf: false,
            has_mesh_indices: false,
            has_prepared_mesh_2d: false,
            all_2d_analytic: true,
            has_opaque_meshes: false,
            has_transparent_meshes: false,
            grid_3d_draw_calls: 0,
            outputs: RenderOutputs::CPU_RGBA_ONLY,
            background_color: [0.0; 4],
            camera_clip: None,
            camera_raster_scale: 2.0,
        }
    }

    #[test]
    fn empty_scene_selects_an_empty_plan() {
        let plan = RecordingPlan::new(input());
        assert_eq!(plan.execution, FrameExecutionPlan::Empty);
        assert!(!plan.runs_postprocess);
        assert!(!plan.has_surface_overlay);
        assert_eq!(plan.raster_scale, 2);
        let stats = plan.stats(&[], 0, 0, &[], 0, RenderOutputs::CPU_RGBA_ONLY);
        assert_eq!(stats.sdf_dispatches, 0);
        assert_eq!(stats.raster_passes, 0);
        assert_eq!(stats.tone_map_dispatches, 0);
        assert_eq!(stats.rgba_readback_copies, 1);
    }

    #[test]
    fn sdf_only_runs_surface_lighting_without_raster() {
        let plan = RecordingPlan::new(RecordingPlanInput {
            has_sdf: true,
            ..input()
        });
        assert_eq!(plan.execution, FrameExecutionPlan::SdfOnly);
        assert!(plan.runs_postprocess);
        assert!(!plan.has_surface_overlay);
        let stats = plan.stats(&[], 0, 0, &[], 0, RenderOutputs::CPU_RGBA_ONLY);
        assert_eq!(stats.sdf_dispatches, 1);
        assert_eq!(stats.surface_lighting_dispatches, 1);
        assert_eq!(stats.surface_composite_dispatches, 1);
        assert_eq!(stats.raster_passes, 0);
    }

    #[test]
    fn analytic_2d_forces_output_resolution_tone_map() {
        let plan = RecordingPlan::new(RecordingPlanInput {
            has_prepared_mesh_2d: true,
            ..input()
        });
        assert!(plan.analytic_2d);
        assert_eq!(plan.execution, FrameExecutionPlan::RasterToneMap);
        assert_eq!(plan.raster_scale, 1);
        assert!(!plan.fused_video_downsample);
        let stats = plan.stats(&[], 1, 4, &[], 0, RenderOutputs::CPU_RGBA_ONLY);
        assert_eq!(stats.mesh_2d_analytic_aa, 1);
        assert_eq!(stats.downsample_dispatches, 0);
        assert_eq!(stats.tone_map_dispatches, 1);
    }

    #[test]
    fn bloom_disables_analytic_2d_and_keeps_ssaa() {
        let plan = RecordingPlan::new(RecordingPlanInput {
            has_prepared_mesh_2d: true,
            bloom_enabled: true,
            ..input()
        });
        assert!(!plan.analytic_2d);
        assert_eq!(plan.execution, FrameExecutionPlan::RasterDownsample);
        assert_eq!(plan.raster_scale, 2);
        assert_eq!(
            plan.stats(&[], 1, 1, &[], 0, RenderOutputs::NONE)
                .bloom_dispatches,
            3
        );
    }

    #[test]
    fn fused_video_downsample_requires_ssaa_raster_without_cpu_outputs() {
        let plan = RecordingPlan::new(RecordingPlanInput {
            has_mesh_indices: true,
            has_opaque_meshes: true,
            analytic_aa_2d: false,
            outputs: RenderOutputs::VULKAN_VIDEO_ONLY,
            ..input()
        });
        assert_eq!(plan.execution, FrameExecutionPlan::RasterDownsample);
        assert!(plan.fused_video_downsample);
        assert!(!plan.runs_postprocess);
        assert!(plan.uses_deferred_raster);
        let stats = plan.stats(
            &[Mesh3DDraw {
                first_index: 0,
                index_count: 36,
                material_index: 0,
                transparent: false,
                view_depth: 1.0,
            }],
            0,
            0,
            &[],
            0,
            RenderOutputs::VULKAN_VIDEO_ONLY,
        );
        assert_eq!(stats.fused_video_downsample_dispatches, 1);
        assert_eq!(stats.downsample_dispatches, 0);
        assert_eq!(stats.tone_map_dispatches, 0);
        assert_eq!(stats.mesh_3d_opaque_draw_calls, 1);
    }

    #[test]
    fn sdf_plus_2d_requests_a_surface_overlay() {
        let plan = RecordingPlan::new(RecordingPlanInput {
            has_sdf: true,
            has_prepared_mesh_2d: true,
            ..input()
        });
        assert_eq!(plan.execution, FrameExecutionPlan::SdfRasterComposite);
        assert!(plan.has_surface_overlay);
        assert!(!plan.uses_deferred_raster);
        assert_eq!(
            plan.stats(&[], 1, 1, &[], 0, RenderOutputs::CPU_RGBA_ONLY)
                .raster_passes,
            1
        );
    }

    #[test]
    fn grid_only_uses_one_depth_tested_raster_pass() {
        let plan = RecordingPlan::new(RecordingPlanInput {
            grid_3d_draw_calls: 2,
            ..input()
        });
        assert_eq!(plan.execution, FrameExecutionPlan::RasterDownsample);
        assert!(plan.raster_uses_depth);
        assert!(!plan.has_surface_overlay);
        let stats = plan.stats(&[], 0, 0, &[], 0, RenderOutputs::CPU_RGBA_ONLY);
        assert_eq!(stats.grid_3d_draw_calls, 2);
        assert_eq!(stats.raster_passes, 1);
        assert_eq!(stats.depth_attachment_raster_passes, 1);
    }

    #[test]
    fn sdf_plus_grid_routes_grid_through_surface_overlay() {
        let plan = RecordingPlan::new(RecordingPlanInput {
            has_sdf: true,
            grid_3d_draw_calls: 1,
            ..input()
        });
        assert_eq!(plan.execution, FrameExecutionPlan::SdfRasterComposite);
        assert!(plan.has_surface_overlay);
        assert!(!plan.uses_deferred_raster);
    }

    #[test]
    fn transparent_deferred_meshes_count_extra_raster_passes() {
        let plan = RecordingPlan::new(RecordingPlanInput {
            has_mesh_indices: true,
            has_opaque_meshes: true,
            has_transparent_meshes: true,
            ssaa_factor: 1,
            ..input()
        });
        assert!(plan.uses_deferred_raster);
        assert!(plan.has_surface_overlay);
        let draws = [
            Mesh3DDraw {
                first_index: 0,
                index_count: 36,
                material_index: 0,
                transparent: false,
                view_depth: 1.0,
            },
            Mesh3DDraw {
                first_index: 36,
                index_count: 36,
                material_index: 1,
                transparent: true,
                view_depth: 0.5,
            },
        ];
        let stats = plan.stats(&draws, 0, 0, &[], 0, RenderOutputs::CPU_RGBA_ONLY);
        assert_eq!(stats.mesh_3d_transparent_draw_calls, 2);
        assert_eq!(stats.raster_passes, 3);
        assert_eq!(stats.depth_attachment_raster_passes, 2);
    }
}
