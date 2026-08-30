use crate::mobjects::mesh_3d::{AlphaMode3D, SurfaceMaterial};
use crate::video_backend::vulkan_h264::VulkanVideoFrame;
use crate::vulkan::context::VulkanContext;
use ash::vk;
use std::sync::Arc;

mod frame;
mod mesh_2d;
mod output;
mod pipelines;
mod prepared_frame;
mod profiling;
mod record;
mod resource;
mod scene;
mod targets;
mod textures;
mod upload;
mod video_output;
use frame::{FrameProfile, FrameSet, RENDER_FRAME_COUNT};
use mesh_2d::{Mesh2DBatch, Mesh2DUploadPlanner, PrepareMesh2DError, PreparedMesh2D};
use output::OutputRetirement;
pub use output::RenderOutputs;
use pipelines::PipelineSet;
use prepared_frame::{FrameOptions, FrameRequirements, PreparedFrame};
pub use profiling::{GpuPassTimings, RendererStats};
use record::{CommandRecorder, FrameRecord, GeometryUploadBuffers2D};
use resource::DescriptorPool;
pub use resource::{Buffer, Image};
use scene::{PreparedScene, ScenePreparer};
use targets::{TargetCache, TargetCacheResources};
use textures::{StudioEnvironment, TextureRegistry};
use upload::{FrameBuffers, FrameUpload};

const MAX_SURFACE_MATERIALS: usize = 10_000;
const MAX_GRIDS_3D: usize = 1_024;

fn align_up(value: u64, alignment: u64) -> u64 {
    if alignment <= 1 {
        value
    } else {
        (value + alignment - 1) & !(alignment - 1)
    }
}

fn msaa_to_vk_sample_count(msaa_samples: u32) -> vk::SampleCountFlags {
    match msaa_samples {
        1 => vk::SampleCountFlags::TYPE_1,
        2 => vk::SampleCountFlags::TYPE_2,
        4 => vk::SampleCountFlags::TYPE_4,
        8 => vk::SampleCountFlags::TYPE_8,
        16 => vk::SampleCountFlags::TYPE_16,
        32 => vk::SampleCountFlags::TYPE_32,
        64 => vk::SampleCountFlags::TYPE_64,
        _ => vk::SampleCountFlags::TYPE_8,
    }
}

fn get_max_usable_sample_count(
    ctx: &VulkanContext,
    requested: vk::SampleCountFlags,
) -> vk::SampleCountFlags {
    let properties = unsafe {
        ctx.instance
            .get_physical_device_properties(ctx.physical_device)
    };
    let counts = properties.limits.framebuffer_color_sample_counts
        & properties.limits.framebuffer_depth_sample_counts;

    if counts.contains(requested) {
        return requested;
    }

    // Fallback to highest supported
    let fallbacks = [
        vk::SampleCountFlags::TYPE_64,
        vk::SampleCountFlags::TYPE_32,
        vk::SampleCountFlags::TYPE_16,
        vk::SampleCountFlags::TYPE_8,
        vk::SampleCountFlags::TYPE_4,
        vk::SampleCountFlags::TYPE_2,
        vk::SampleCountFlags::TYPE_1,
    ];

    for &fallback in &fallbacks {
        if counts.contains(fallback) && fallback.as_raw() <= requested.as_raw() {
            return fallback;
        }
    }
    vk::SampleCountFlags::TYPE_1
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Nv12Constants {
    pub width: u32,
    pub height: u32,
    pub _padding: [u32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ToneMapConstants {
    /// Raster-scale factor of the resolved image relative to the output.
    /// Analytic-AA 2D frames raster at 1x even when `ssaa_factor` is larger,
    /// so the tone-map downsample loop must use this instead of the image
    /// dimensions.
    pub factor: u32,
    pub _padding: [u32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform2D {
    pub width: f32,
    pub height: f32,
    pub scale_factor: f32,
    pub _pad: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub pos: [f32; 3],
    pub _padding0: u32,
    pub look_at: [f32; 3],
    pub _padding1: u32,
    pub up: [f32; 3],
    pub fov: f32,
    pub width: f32,
    pub height: f32,
    pub proj_type: u32,
    pub ortho_left: f32,
    pub ortho_right: f32,
    pub ortho_bottom: f32,
    pub ortho_top: f32,
    pub has_clip: u32,
    pub clip_x: f32,
    pub clip_y: f32,
    pub clip_w: f32,
    pub clip_h: f32,
    pub aa_level: u32,
    pub num_primitives: u32,
    pub raster_scale: u32,
    pub has_raster_surfaces: u32,
    pub proj_mat: [f32; 16],
    pub light_pos: [f32; 3],
    pub light_intensity: f32,
    pub light_color: [f32; 3],
    pub environment_intensity: f32,
    pub environment_color: [f32; 3],
    pub environment_rotation: f32,
    pub background_color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct MaterialData3D {
    base_color: [f32; 4],
    emissive: [f32; 4],
    grid_color: [f32; 4],
    surface: [f32; 4],
    grid: [f32; 4],
    grid_backface: [f32; 4],
    transmission: [f32; 4],
    absorption: [f32; 4],
    patch_corner_0: [f32; 4],
    patch_corner_1: [f32; 4],
    patch_corner_2: [f32; 4],
    patch_color: [f32; 4],
    patch_edge_color: [f32; 4],
    patch_params: [f32; 4],
}

impl From<SurfaceMaterial> for MaterialData3D {
    fn from(material: SurfaceMaterial) -> Self {
        let grid = material.spherical_grid.unwrap_or_default();
        let patch = material.spherical_patch;
        let transmission = match material.alpha_mode {
            AlphaMode3D::Opaque => None,
            AlphaMode3D::Blend(transmission) => Some(transmission),
        };
        let patch_directions = patch.map(|patch| patch.directions).unwrap_or([[0.0; 3]; 3]);

        let patch_color = patch.map(|patch| patch.color).unwrap_or([0.0; 4]);
        let patch_edge_color = patch.map(|patch| patch.edge_color).unwrap_or([0.0; 4]);
        let patch_corner_0 = [
            patch_directions[0][0],
            patch_directions[0][1],
            patch_directions[0][2],
            0.0,
        ];
        let patch_params = [
            patch
                .map(|patch| patch.edge_width_pixels)
                .unwrap_or_default(),
            0.0,
            0.0,
            0.0,
        ];

        Self {
            base_color: material.base_color,
            emissive: [
                material.emissive[0],
                material.emissive[1],
                material.emissive[2],
                material.emissive_strength,
            ],
            grid_color: grid.color,
            surface: [
                material.roughness,
                material.metallic,
                material.reflectance,
                if transmission.is_some() { 1.0 } else { 0.0 },
            ],
            grid: [
                grid.longitude_count,
                grid.latitude_count,
                grid.line_width_pixels,
                if material.spherical_grid.is_some() {
                    1.0
                } else {
                    0.0
                },
            ],
            grid_backface: [
                grid.backface_intensity,
                if material.unlit { 1.0 } else { 0.0 },
                if material.flat_shading { 1.0 } else { 0.0 },
                0.0,
            ],
            transmission: transmission
                .map(|transmission| {
                    [
                        transmission.opacity,
                        transmission.fresnel_opacity,
                        transmission.ior,
                        0.0,
                    ]
                })
                .unwrap_or([1.0, 0.0, 1.0, 0.0]),
            absorption: transmission
                .map(|transmission| {
                    [
                        transmission.absorption[0],
                        transmission.absorption[1],
                        transmission.absorption[2],
                        transmission.backface_opacity_scale,
                    ]
                })
                .unwrap_or([0.0, 0.0, 0.0, 1.0]),
            patch_corner_0,
            patch_corner_1: [
                patch_directions[1][0],
                patch_directions[1][1],
                patch_directions[1][2],
                0.0,
            ],
            patch_corner_2: [
                patch_directions[2][0],
                patch_directions[2][1],
                patch_directions[2][2],
                0.0,
            ],
            patch_color,
            patch_edge_color,
            patch_params,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Mesh3DDraw {
    first_index: u32,
    index_count: u32,
    material_index: u32,
    transparent: bool,
    view_depth: f32,
}

impl Mesh3DDraw {
    fn is_transparent(self) -> bool {
        self.transparent
    }
}

pub struct VulkanRenderer {
    // Fields with Vulkan handles are declared in teardown order. Rust drops fields
    // top-to-bottom after `Drop::drop`; `cache` is taken explicitly there first.
    cache: Option<TargetCache>,
    frames: FrameSet,
    output_retirement: OutputRetirement,
    textures: TextureRegistry,
    environment: StudioEnvironment,
    frame_buffers: FrameBuffers,
    pipelines: PipelineSet,
    descriptor_pool: Arc<DescriptorPool>,
    ctx: Arc<VulkanContext>,

    mesh_upload_planner_2d: Mesh2DUploadPlanner,
    scene_preparer: ScenePreparer,
    last_stats: RendererStats,
    gpu_profiling: bool,
    last_gpu_timings: Option<GpuPassTimings>,
    bloom_enabled: bool,
    analytic_aa_2d: bool,
    msaa_samples: u32,
    ssaa_factor: u32,
}

impl VulkanRenderer {
    pub fn new(ctx: Arc<VulkanContext>, config: crate::RendererConfig) -> Self {
        let msaa_samples = config.msaa_samples;
        let ssaa_factor = config.ssaa_factor;
        let analytic_aa_2d = config.analytic_aa_2d;
        let output_transform = match config.output_color_profile {
            crate::OutputColorProfile::Bt709Sdr => {
                include_str!("output_transform_bt709.wgsl")
            }
            crate::OutputColorProfile::Bt2020Pq | crate::OutputColorProfile::Bt2020Hlg => {
                panic!(
                    "{:?} requires a 10-bit HDR output path; the current renderer outputs RGBA8/NV12",
                    config.output_color_profile
                )
            }
        };
        let requested_sample_count = msaa_to_vk_sample_count(msaa_samples);
        let sample_count = get_max_usable_sample_count(&ctx, requested_sample_count);
        let pipelines = PipelineSet::new(&ctx, output_transform, sample_count);

        let descriptor_pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 96,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLED_IMAGE,
                descriptor_count: 96,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLER,
                descriptor_count: 8,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 40,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 40,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
                descriptor_count: 20,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER_DYNAMIC,
                descriptor_count: 20,
            },
        ];
        let descriptor_pool = Arc::new(DescriptorPool::new(&ctx, &descriptor_pool_sizes, 48));

        let environment = StudioEnvironment::new(&ctx);
        let textures = TextureRegistry::new(&ctx, &descriptor_pool, &pipelines);

        let frame_buffers = FrameBuffers::new(&ctx);
        let (static_vertex_buffer_2d_size, static_index_buffer_2d_size) =
            frame_buffers.static_mesh_2d_capacities();
        let frames = FrameSet::new(Arc::clone(&ctx));
        let output_retirement = OutputRetirement::new(Arc::clone(&ctx));

        Self {
            ctx,
            environment,
            descriptor_pool,
            pipelines,
            frame_buffers,
            textures,
            mesh_upload_planner_2d: Mesh2DUploadPlanner::new(
                static_vertex_buffer_2d_size,
                static_index_buffer_2d_size,
            ),
            scene_preparer: ScenePreparer::default(),
            last_stats: RendererStats::default(),
            gpu_profiling: false,
            last_gpu_timings: None,
            bloom_enabled: false,
            analytic_aa_2d,
            frames,
            output_retirement,
            cache: None,
            msaa_samples: sample_count.as_raw(),
            ssaa_factor,
        }
    }

    pub fn last_stats(&self) -> RendererStats {
        self.last_stats
    }

    pub fn set_gpu_profiling(&mut self, enabled: bool) {
        self.gpu_profiling = enabled;
        if !enabled {
            self.last_gpu_timings = None;
        }
    }

    pub fn set_bloom_enabled(&mut self, enabled: bool) {
        self.bloom_enabled = enabled;
    }

    pub fn last_gpu_timings(&self) -> Option<GpuPassTimings> {
        self.last_gpu_timings
    }

    pub fn render_scene(
        &mut self,
        scene: &crate::Scene,
        scene_config: &crate::SceneConfig,
        output: Option<&mut [u8]>,
    ) {
        self.render_scene_with_outputs(scene, scene_config, output, RenderOutputs::CPU_READBACKS);
    }

    pub fn render_scene_with_outputs(
        &mut self,
        scene: &crate::Scene,
        scene_config: &crate::SceneConfig,
        output: Option<&mut [u8]>,
        mut outputs: RenderOutputs,
    ) {
        outputs.cpu_rgba |= output.is_some();
        let prepared = self
            .scene_preparer
            .prepare(scene, scene_config, self.ssaa_factor);
        self.render(prepared, output, outputs);
    }

    pub fn update_texture(&mut self, index: u32, width: u32, height: u32, data: &[u8]) {
        self.textures.update(index, width, height, data);
    }

    pub fn current_output_image_view(&self) -> Option<vk::ImageView> {
        let cache = self.cache.as_ref()?;
        self.output_retirement.current_image_view(cache)
    }

    pub fn bind_texture_view(&mut self, index: u32, view: vk::ImageView) {
        self.textures.bind(index, view);
    }

    pub fn wait_idle(&self) {
        unsafe {
            let _ = self.ctx.device.device_wait_idle();
        }
    }

    fn render(&mut self, scene: PreparedScene, output: Option<&mut [u8]>, outputs: RenderOutputs) {
        let requirements = FrameRequirements::for_scene(&scene);
        self.ensure_target_cache(requirements);

        let frame_index = self.cache.as_ref().unwrap().current_frame % RENDER_FRAME_COUNT;
        let (mesh_2d, mesh_2d_arena_rebuilds) =
            self.prepare_mesh_2d(frame_index, &scene.mesh_batches_2d);
        let frame = PreparedFrame::new(
            scene,
            requirements,
            mesh_2d,
            mesh_2d_arena_rebuilds,
            FrameOptions {
                ssaa_factor: self.ssaa_factor,
                analytic_aa_2d: self.analytic_aa_2d,
                bloom_enabled: self.bloom_enabled,
                gpu_profiling: self.gpu_profiling,
                outputs,
            },
        );
        self.submit_prepared_frame(frame, output);
    }

    fn ensure_target_cache(&mut self, requirements: FrameRequirements) {
        let cache_needs_update = self
            .cache
            .as_ref()
            .is_none_or(|cache| !cache.satisfies(requirements));
        if !cache_needs_update {
            return;
        }

        if let Some(old_cache) = self.cache.take() {
            unsafe {
                self.ctx.device.device_wait_idle().unwrap();
            }
            drop(old_cache);
        }
        let resources = TargetCacheResources {
            ctx: &self.ctx,
            descriptor_pool: &self.descriptor_pool,
            pipelines: &self.pipelines,
            msaa_samples: self.msaa_samples,
            ssaa_factor: self.ssaa_factor,
            environment_map: &self.environment.image,
            environment_sampler: self.environment.sampler,
            camera_buffer: &self.frame_buffers.camera,
            material_buffer_3d: &self.frame_buffers.material_3d,
            primitive_buffer: &self.frame_buffers.primitive,
            grid_buffer_3d: &self.frame_buffers.grid_3d,
            camera_buffer_2d: &self.frame_buffers.camera_2d,
            nv12_constants_buffer: &self.frame_buffers.nv12_constants,
            tone_map_factor_buffer: &self.frame_buffers.tone_map_factor,
            camera_buffer_stride: self.frame_buffers.strides.camera,
            material_buffer_3d_stride: self.frame_buffers.strides.material_3d,
            primitive_buffer_stride: self.frame_buffers.strides.primitive,
            grid_buffer_3d_stride: self.frame_buffers.strides.grid_3d,
            camera_buffer_2d_stride: self.frame_buffers.strides.camera_2d,
            tone_map_factor_stride: self.frame_buffers.strides.tone_map_factor,
        };
        self.cache = Some(TargetCache::new(requirements, &resources));
    }

    fn prepare_mesh_2d(
        &mut self,
        frame_index: usize,
        mesh_batches: &[Mesh2DBatch],
    ) -> (PreparedMesh2D, u32) {
        let arenas = self.mesh_upload_planner_2d.frame_arenas(
            frame_index as u64,
            self.frame_buffers.strides.vertex_staging_2d,
            self.frame_buffers.strides.index_staging_2d,
            self.frame_buffers.strides.instance_2d,
        );
        match self.mesh_upload_planner_2d.prepare(arenas, mesh_batches) {
            Ok(prepared) => (prepared, 0),
            Err(PrepareMesh2DError::StaticGeometry) => {
                unsafe {
                    self.ctx.device.device_wait_idle().unwrap();
                }
                self.mesh_upload_planner_2d.reset_static_arena();
                let prepared = self
                    .mesh_upload_planner_2d
                    .prepare(arenas, mesh_batches)
                    .expect("active 2D scene exceeds a frame or persistent geometry arena");
                (prepared, 1)
            }
            Err(error) => panic!("2D frame preparation failed: {error:?}"),
        }
    }

    fn submit_prepared_frame(&mut self, prepared: PreparedFrame, output: Option<&mut [u8]>) {
        let PreparedFrame {
            scene,
            sdf_primitives,
            grids_3d,
            mesh_2d,
            plan,
            outputs,
            stats,
        } = prepared;
        let PreparedMesh2D {
            batches: mesh_batches_2d,
            uploads: geometry_uploads_2d,
            instances: instances_2d,
        } = mesh_2d;

        let cache = self.cache.as_mut().unwrap();
        cache.ensure_frame_attachments(&self.ctx, plan, self.msaa_samples);
        let output_submission = self
            .output_retirement
            .prepare_submission(cache, plan, outputs);
        let frame_index = output_submission.frame_index();
        let video_frame_index = output_submission.video_frame_index();
        self.last_stats = stats;

        let (frame, completed_timings) = self.frames.begin(frame_index, plan.gpu_profiling);
        if let Some(timings) = completed_timings {
            self.last_gpu_timings = Some(timings);
        }
        let command_buffer = frame.command_buffer();
        let query_pool = frame.query_pool();
        let uploaded = self.frame_buffers.upload(
            frame_index,
            FrameUpload {
                camera: &scene.camera_uniform,
                camera_2d: &scene.camera_uniform_2d,
                primitives: &sdf_primitives,
                grids_3d: &grids_3d,
                materials: &scene.surface_materials,
                mesh_vertices: &scene.mesh_vertices,
                mesh_indices: &scene.mesh_indices,
                geometry_2d: &geometry_uploads_2d,
                instances_2d: &instances_2d,
                tone_map_factor: plan.raster_scale,
            },
        );

        unsafe {
            let recorder = CommandRecorder::new(&self.ctx.device, command_buffer, &self.pipelines);
            recorder.record_frame(FrameRecord {
                plan,
                cache,
                frame_index,
                video_frame_index,
                uploaded,
                outputs,
                mesh_draws_3d: &scene.mesh_draws_3d,
                grid_count_3d: grids_3d.len() as u32,
                mesh_batches_2d: &mesh_batches_2d,
                geometry_uploads_2d: &geometry_uploads_2d,
                uploads_2d: GeometryUploadBuffers2D {
                    vertex_staging: self.frame_buffers.vertex_staging_2d.vk_buffer,
                    vertex_staging_base: uploaded.vertex_staging_2d_offset,
                    index_staging: self.frame_buffers.index_staging_2d.vk_buffer,
                    index_staging_base: uploaded.index_staging_2d_offset,
                    vertex_device: self.frame_buffers.vertex_2d.vk_buffer,
                    index_device: self.frame_buffers.index_2d.vk_buffer,
                },
                mesh_3d_vertex: self.frame_buffers.vertex.vk_buffer,
                mesh_3d_index: self.frame_buffers.index.vk_buffer,
                mesh_2d_vertex: self.frame_buffers.vertex_2d.vk_buffer,
                mesh_2d_index: self.frame_buffers.index_2d.vk_buffer,
                mesh_2d_instance: self.frame_buffers.instance_2d.vk_buffer,
                raster_texture_set: self.textures.descriptor_set(),
                query_pool,
            });

            frame.submit(
                output_submission.wait_infos(),
                output_submission.signal_infos(),
                plan.gpu_profiling.then_some(FrameProfile {
                    plan: plan.execution,
                    geometry_upload: !geometry_uploads_2d.is_empty(),
                    postprocess: plan.runs_postprocess,
                    output: outputs.cpu_nv12
                        || outputs.cpu_yuv444p
                        || outputs.vulkan_video
                        || outputs.cpu_rgba,
                }),
            );
        }

        if let Some(out_buf) = output {
            self.output_retirement
                .copy_submitted_rgba(cache, &self.frames, frame_index, out_buf);
        }
        output_submission.commit(cache);
    }

    pub fn get_nv12_bytes(&self) -> Option<&[u8]> {
        let cache = self.cache.as_ref()?;
        self.output_retirement.latest_nv12(cache, &self.frames)
    }

    pub fn get_yuv444p_bytes(&self) -> Option<&[u8]> {
        let cache = self.cache.as_ref()?;
        self.output_retirement.latest_yuv444p(cache, &self.frames)
    }

    pub fn get_vulkan_video_frame(&mut self) -> Option<VulkanVideoFrame> {
        let cache = self.cache.as_mut()?;
        self.output_retirement.take_video_frame(cache)
    }

    pub fn get_rgba_bytes(&mut self) -> Option<&[u8]> {
        let cache = self.cache.as_ref()?;
        self.output_retirement.latest_rgba(cache, &self.frames)
    }
}

impl Drop for VulkanRenderer {
    fn drop(&mut self) {
        unsafe {
            let _ = self.ctx.device.device_wait_idle();
        }
        drop(self.cache.take());
    }
}

#[cfg(test)]
mod tests {
    use super::profiling::timestamp_delta;
    use nalgebra::{Point3, Vector3};

    use super::MaterialData3D;
    use super::frame::FrameExecutionPlan;
    use super::prepared_frame::GpuSdfPrimitive;
    use crate::mobjects::mesh_3d::{
        AlphaMode3D, SphericalPatchMaterial, SurfaceMaterial, Transmission3D,
    };
    use crate::mobjects::object_3d::SdfPrimitive;

    #[test]
    fn frame_execution_plan_eliminates_empty_passes() {
        assert_eq!(
            FrameExecutionPlan::build(false, false, 2),
            FrameExecutionPlan::Empty
        );
        assert_eq!(
            FrameExecutionPlan::build(true, false, 2),
            FrameExecutionPlan::SdfOnly
        );
        assert_eq!(
            FrameExecutionPlan::build(false, true, 1),
            FrameExecutionPlan::RasterToneMap
        );
        assert_eq!(
            FrameExecutionPlan::build(false, true, 2),
            FrameExecutionPlan::RasterDownsample
        );
        assert_eq!(
            FrameExecutionPlan::build(true, true, 1),
            FrameExecutionPlan::SdfRasterComposite
        );
        assert_eq!(
            FrameExecutionPlan::build(true, true, 2),
            FrameExecutionPlan::SdfRasterComposite
        );
    }

    #[test]
    fn sdf_primitives_encode_to_the_shader_abi() {
        assert_eq!(std::mem::size_of::<GpuSdfPrimitive>(), 64);

        let primitives = [
            SdfPrimitive::Sphere {
                center: Point3::new(1.0, 2.0, 3.0),
                radius: 4.0,
            },
            SdfPrimitive::Capsule {
                start: Point3::origin(),
                end: Point3::new(1.0, 2.0, 3.0),
                radius: 0.5,
            },
            SdfPrimitive::Arrow {
                start: Point3::origin(),
                end: Point3::new(1.0, 0.0, 0.0),
                shaft_radius: 0.1,
                head_radius: 0.2,
                head_length: 0.3,
            },
            SdfPrimitive::OrientedBox {
                center: Point3::new(1.0, 2.0, 3.0),
                half_extents: Vector3::new(4.0, 5.0, 6.0),
                x_axis: Vector3::new(1.0, 0.0, 0.0),
                y_axis: Vector3::new(0.0, 1.0, 0.0),
            },
            SdfPrimitive::QuadraticBezier {
                start: Point3::origin(),
                control: Point3::new(1.0, 2.0, 3.0),
                end: Point3::new(4.0, 5.0, 6.0),
                radius: 0.25,
            },
        ];

        for (shape_type, primitive) in primitives.into_iter().enumerate() {
            let encoded = GpuSdfPrimitive::encode(primitive, 17);
            assert_eq!(encoded.material_index, 17);
            assert_eq!(encoded.shape_type, shape_type as u32);
            assert_eq!(encoded.padding, [0; 2]);
        }

        let encoded = GpuSdfPrimitive::encode(primitives[3], 0);
        assert_eq!(
            encoded.params,
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,]
        );
    }

    #[test]
    fn spherical_patch_material_has_a_stable_gpu_layout() {
        let material = SurfaceMaterial {
            alpha_mode: AlphaMode3D::Blend(Transmission3D {
                opacity: 0.1,
                fresnel_opacity: 0.2,
                absorption: [0.3, 0.4, 0.5],
                ior: 1.7,
                backface_opacity_scale: 0.8,
            }),
            spherical_patch: Some(SphericalPatchMaterial {
                directions: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                color: [0.1, 0.2, 0.3, 0.4],
                edge_color: [0.5, 0.6, 0.7, 0.8],
                edge_width_pixels: 2.25,
            }),
            ..Default::default()
        };
        let gpu = MaterialData3D::from(material);

        assert_eq!(std::mem::size_of::<MaterialData3D>(), 14 * 16);
        assert_eq!(gpu.patch_corner_0, [1.0, 2.0, 3.0, 0.0]);
        assert_eq!(gpu.patch_corner_2, [7.0, 8.0, 9.0, 0.0]);
        assert_eq!(gpu.patch_color, [0.1, 0.2, 0.3, 0.4]);
        assert_eq!(gpu.patch_edge_color, [0.5, 0.6, 0.7, 0.8]);
        assert_eq!(gpu.patch_params[0], 2.25);
        assert_eq!(gpu.transmission, [0.1, 0.2, 1.7, 0.0]);
        assert_eq!(gpu.absorption, [0.3, 0.4, 0.5, 0.8]);
    }

    #[test]
    fn timestamp_delta_handles_hardware_counter_wraparound() {
        assert_eq!(timestamp_delta(10, 25, 64), 15);
        assert_eq!(timestamp_delta(250, 5, 8), 11);
    }
}
