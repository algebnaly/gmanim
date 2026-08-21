use ash::vk;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum FrameExecutionPlan {
    Empty,
    SdfOnly,
    RasterToneMap,
    RasterDownsample,
    SdfRasterComposite,
}

impl FrameExecutionPlan {
    pub(super) fn build(has_sdf: bool, has_raster: bool, ssaa_factor: u32) -> Self {
        match (has_sdf, has_raster, ssaa_factor) {
            (false, false, _) => Self::Empty,
            (true, false, _) => Self::SdfOnly,
            (false, true, 1) => Self::RasterToneMap,
            (false, true, _) => Self::RasterDownsample,
            (true, true, _) => Self::SdfRasterComposite,
        }
    }

    pub(super) fn runs_sdf(self) -> bool {
        matches!(self, Self::SdfOnly | Self::SdfRasterComposite)
    }

    pub(super) fn runs_raster(self) -> bool {
        matches!(
            self,
            Self::RasterToneMap | Self::RasterDownsample | Self::SdfRasterComposite
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct TrackedImageState {
    pub(super) layout: vk::ImageLayout,
    pub(super) stage: vk::PipelineStageFlags2,
    pub(super) access: vk::AccessFlags2,
}

impl TrackedImageState {
    pub(super) const UNDEFINED: Self = Self {
        layout: vk::ImageLayout::UNDEFINED,
        stage: vk::PipelineStageFlags2::NONE,
        access: vk::AccessFlags2::NONE,
    };
}

pub(super) unsafe fn transition_image(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    image: vk::Image,
    aspect_mask: vk::ImageAspectFlags,
    state: &mut TrackedImageState,
    next: TrackedImageState,
) {
    let barrier = vk::ImageMemoryBarrier2::default()
        .src_stage_mask(state.stage)
        .src_access_mask(state.access)
        .dst_stage_mask(next.stage)
        .dst_access_mask(next.access)
        .old_layout(state.layout)
        .new_layout(next.layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });
    let dependency =
        vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&barrier));
    unsafe {
        device.cmd_pipeline_barrier2(command_buffer, &dependency);
    }
    *state = next;
}

pub(super) unsafe fn write_gpu_timestamp(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    query_pool: vk::QueryPool,
    query: u32,
    enabled: bool,
) {
    if enabled {
        unsafe {
            device.cmd_write_timestamp2(
                command_buffer,
                vk::PipelineStageFlags2::ALL_COMMANDS,
                query_pool,
                query,
            );
        }
    }
}
