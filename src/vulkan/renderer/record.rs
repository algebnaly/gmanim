use ash::vk;

use super::Mesh3DDraw;
use super::frame::TrackedImageState;
use super::mesh_2d::PreparedMesh2DBatch;
use super::pipelines::PipelineSet;
use super::prepared_frame::GpuGrid3D;

mod frame;
mod output;
mod plan;
mod postprocess;
mod raster;
mod sdf;
mod upload;

pub(super) use frame::FrameRecord;
pub(super) use plan::{RecordingPlan, RecordingPlanInput};

pub(super) struct CommandRecorder<'a> {
    device: &'a ash::Device,
    command_buffer: vk::CommandBuffer,
    pipelines: &'a PipelineSet,
}

#[derive(Clone, Copy)]
pub(super) struct VideoOutputPass {
    pub(super) image: vk::Image,
    pub(super) descriptor_set: vk::DescriptorSet,
    pub(super) current_layout: vk::ImageLayout,
}

#[derive(Clone, Copy)]
pub(super) struct OutputPasses {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) fused_video_downsample: bool,
    pub(super) cpu_nv12_descriptor_set: Option<vk::DescriptorSet>,
    pub(super) cpu_yuv444p_descriptor_set: Option<vk::DescriptorSet>,
    pub(super) video: Option<VideoOutputPass>,
    pub(super) rgba_buffer: Option<vk::Buffer>,
    pub(super) rgba_padded_bytes_per_row: u32,
}

#[derive(Clone, Copy)]
pub(super) struct GeometryUploadBuffers2D {
    pub(super) vertex_staging: vk::Buffer,
    pub(super) vertex_staging_base: u64,
    pub(super) index_staging: vk::Buffer,
    pub(super) index_staging_base: u64,
    pub(super) vertex_device: vk::Buffer,
    pub(super) index_device: vk::Buffer,
}

#[derive(Clone, Copy)]
pub(super) enum Mesh3DPass {
    Opaque,
    TransparentDepth,
    TransparentColor,
}

#[derive(Clone, Copy)]
pub(super) struct Mesh3DBindings<'a> {
    pub(super) draws: &'a [Mesh3DDraw],
    pub(super) descriptor_set: vk::DescriptorSet,
    pub(super) dynamic_offsets: &'a [u32],
    pub(super) vertex_buffer: vk::Buffer,
    pub(super) vertex_offset: u64,
    pub(super) index_buffer: vk::Buffer,
    pub(super) index_offset: u64,
}

#[derive(Clone, Copy)]
pub(super) struct Grid3DBindings<'a> {
    pub(super) grids: &'a [GpuGrid3D],
    pub(super) raster_scale: u32,
    pub(super) descriptor_set: vk::DescriptorSet,
    pub(super) dynamic_offsets: &'a [u32],
}

#[derive(Clone, Copy)]
pub(super) enum Mesh2DPass {
    Depth,
    Depthless,
    Analytic,
}

#[derive(Clone, Copy)]
pub(super) struct Mesh2DBindings<'a> {
    pub(super) batches: &'a [PreparedMesh2DBatch],
    pub(super) camera_descriptor_set: vk::DescriptorSet,
    pub(super) camera_dynamic_offsets: &'a [u32],
    pub(super) texture_descriptor_set: vk::DescriptorSet,
    pub(super) vertex_buffer: vk::Buffer,
    pub(super) index_buffer: vk::Buffer,
    pub(super) instance_buffer: vk::Buffer,
    pub(super) instance_offset: u64,
}

pub(super) struct RasterAttachment<'a> {
    pub(super) image: vk::Image,
    pub(super) view: vk::ImageView,
    pub(super) state: &'a mut TrackedImageState,
}

pub(super) enum ColorAttachment<'a> {
    Single(RasterAttachment<'a>),
    Resolve {
        multisample: RasterAttachment<'a>,
        resolved: RasterAttachment<'a>,
        preserve_multisample: bool,
    },
}

#[derive(Clone, Copy)]
pub(super) enum ColorLoad {
    Clear([f32; 4]),
    Load,
}

#[derive(Clone, Copy)]
pub(super) enum DepthLoad {
    Clear,
    Load,
    Discard,
}

pub(super) struct DepthAttachment<'a> {
    pub(super) attachment: RasterAttachment<'a>,
    pub(super) load: DepthLoad,
    pub(super) preserve: bool,
}

#[derive(Clone, Copy)]
pub(super) struct RasterRegion {
    extent: vk::Extent2D,
    viewport: vk::Viewport,
    scissor: vk::Rect2D,
}

impl RasterRegion {
    pub(super) fn new(extent: vk::Extent2D, clip: Option<[f32; 4]>, scale: f32) -> Self {
        let [x, y, width, height] = match clip {
            Some([x, y, width, height]) => [x * scale, y * scale, width * scale, height * scale],
            None => [0.0, 0.0, extent.width as f32, extent.height as f32],
        };
        Self {
            extent,
            viewport: vk::Viewport {
                x,
                y,
                width,
                height,
                min_depth: 0.0,
                max_depth: 1.0,
            },
            scissor: vk::Rect2D {
                offset: vk::Offset2D {
                    x: x as i32,
                    y: y as i32,
                },
                extent: vk::Extent2D {
                    width: width as u32,
                    height: height as u32,
                },
            },
        }
    }
}

pub(super) struct DeferredOpaquePass<'a> {
    pub(super) normal_depth: RasterAttachment<'a>,
    pub(super) albedo: RasterAttachment<'a>,
    pub(super) material_id: RasterAttachment<'a>,
    pub(super) depth: RasterAttachment<'a>,
    pub(super) region: RasterRegion,
    pub(super) preserve_depth: bool,
    pub(super) meshes: Mesh3DBindings<'a>,
}

pub(super) struct TransparentDepthPass<'a> {
    pub(super) depth: RasterAttachment<'a>,
    pub(super) extent: vk::Extent2D,
    pub(super) meshes: Mesh3DBindings<'a>,
}

pub(super) struct ColorRasterPass<'a> {
    pub(super) color: ColorAttachment<'a>,
    pub(super) color_load: ColorLoad,
    pub(super) depth: Option<DepthAttachment<'a>>,
    pub(super) region: RasterRegion,
    pub(super) meshes_3d: Option<(Mesh3DPass, Mesh3DBindings<'a>)>,
    pub(super) grids_3d: Option<Grid3DBindings<'a>>,
    pub(super) meshes_2d: Option<(Mesh2DPass, Mesh2DBindings<'a>)>,
}

impl OutputPasses {
    fn has_compute_output(self) -> bool {
        self.cpu_nv12_descriptor_set.is_some()
            || self.cpu_yuv444p_descriptor_set.is_some()
            || self.video.is_some()
    }
}

impl<'a> CommandRecorder<'a> {
    pub(super) fn new(
        device: &'a ash::Device,
        command_buffer: vk::CommandBuffer,
        pipelines: &'a PipelineSet,
    ) -> Self {
        Self {
            device,
            command_buffer,
            pipelines,
        }
    }

    unsafe fn record_compute_dispatch(
        &self,
        pipeline: vk::Pipeline,
        layout: vk::PipelineLayout,
        descriptor_set: vk::DescriptorSet,
        width: u32,
        height: u32,
    ) {
        unsafe {
            self.device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                layout,
                0,
                std::slice::from_ref(&descriptor_set),
                &[],
            );
            self.device.cmd_dispatch(
                self.command_buffer,
                width.div_ceil(16),
                height.div_ceil(16),
                1,
            );
        }
    }

    unsafe fn record_mesh_3d_draw(&self, draw: &Mesh3DDraw) {
        unsafe {
            self.device.cmd_draw_indexed(
                self.command_buffer,
                draw.index_count,
                1,
                draw.first_index,
                0,
                draw.material_index,
            );
        }
    }
}
