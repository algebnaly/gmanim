use ash::vk;

use super::Mesh3DDraw;
use super::mesh_2d::PreparedMesh2DBatch;
use super::pipelines::PipelineSet;
use super::prepared_frame::GpuGrid3D;

mod frame;
pub(super) mod graph;
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
pub(super) struct GeometryUploadBuffers2D {
    pub(super) vertex_staging: vk::Buffer,
    pub(super) vertex_staging_base: u64,
    pub(super) index_staging: vk::Buffer,
    pub(super) index_staging_base: u64,
    pub(super) vertex_device: vk::Buffer,
    pub(super) index_device: vk::Buffer,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
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
    pub(super) descriptor_set: vk::DescriptorSet,
    pub(super) dynamic_offsets: &'a [u32],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
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

#[derive(Clone, Copy)]
pub(super) enum ColorAttachment {
    Single(vk::ImageView),
    Resolve {
        multisample: vk::ImageView,
        resolved: vk::ImageView,
        preserve_multisample: bool,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum ColorLoad {
    Clear([f32; 4]),
    Load,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum DepthLoad {
    Clear,
    Load,
    Discard,
}

#[derive(Clone, Copy)]
pub(super) struct DepthAttachment {
    pub(super) view: vk::ImageView,
    pub(super) load: DepthLoad,
    pub(super) preserve: bool,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct RasterRegion {
    extent: vk::Extent2D,
    viewport: vk::Viewport,
    scissor: vk::Rect2D,
}

impl PartialEq for RasterRegion {
    fn eq(&self, other: &Self) -> bool {
        self.extent == other.extent
            && self.scissor == other.scissor
            && self.viewport.x == other.viewport.x
            && self.viewport.y == other.viewport.y
            && self.viewport.width == other.viewport.width
            && self.viewport.height == other.viewport.height
            && self.viewport.min_depth == other.viewport.min_depth
            && self.viewport.max_depth == other.viewport.max_depth
    }
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
    pub(super) normal_depth: vk::ImageView,
    pub(super) albedo: vk::ImageView,
    pub(super) material_id: vk::ImageView,
    pub(super) depth: vk::ImageView,
    pub(super) region: RasterRegion,
    pub(super) preserve_depth: bool,
    pub(super) meshes: Mesh3DBindings<'a>,
}

pub(super) struct TransparentDepthPass<'a> {
    pub(super) depth: vk::ImageView,
    pub(super) extent: vk::Extent2D,
    pub(super) meshes: Mesh3DBindings<'a>,
}

pub(super) struct ColorRasterPass<'a> {
    pub(super) color: ColorAttachment,
    pub(super) color_load: ColorLoad,
    pub(super) depth: Option<DepthAttachment>,
    pub(super) region: RasterRegion,
    pub(super) meshes_3d: Option<(Mesh3DPass, Mesh3DBindings<'a>)>,
    pub(super) grids_3d: Option<Grid3DBindings<'a>>,
    pub(super) meshes_2d: Option<(Mesh2DPass, Mesh2DBindings<'a>)>,
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
