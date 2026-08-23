use ash::vk;
use std::sync::Arc;

use crate::vulkan::context::VulkanContext;

use super::profiling::GpuPassTimings;

pub(super) const RENDER_FRAME_COUNT: usize = 3;
pub(super) const GPU_TIMESTAMP_COUNT: u32 = 6;

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

#[derive(Clone, Copy)]
pub(super) struct FrameProfile {
    pub(super) plan: FrameExecutionPlan,
    pub(super) geometry_upload: bool,
    pub(super) postprocess: bool,
    pub(super) output: bool,
}

struct FrameSlot {
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    query_pool: vk::QueryPool,
    timestamps_pending: bool,
    profile: FrameProfile,
}

pub(super) struct FrameSet {
    ctx: Arc<VulkanContext>,
    slots: [FrameSlot; RENDER_FRAME_COUNT],
}

pub(super) struct FrameRecording<'a> {
    ctx: &'a VulkanContext,
    slot: &'a mut FrameSlot,
}

impl FrameSet {
    pub(super) fn new(ctx: Arc<VulkanContext>) -> Self {
        let command_pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(ctx.queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let slots = std::array::from_fn(|_| {
            let command_pool = unsafe {
                ctx.device
                    .create_command_pool(&command_pool_info, None)
                    .unwrap()
            };
            let command_buffer = unsafe {
                ctx.device
                    .allocate_command_buffers(
                        &vk::CommandBufferAllocateInfo::default()
                            .command_pool(command_pool)
                            .level(vk::CommandBufferLevel::PRIMARY)
                            .command_buffer_count(1),
                    )
                    .unwrap()[0]
            };
            let fence = unsafe {
                ctx.device
                    .create_fence(
                        &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                        None,
                    )
                    .unwrap()
            };
            let query_pool = unsafe {
                ctx.device
                    .create_query_pool(
                        &vk::QueryPoolCreateInfo::default()
                            .query_type(vk::QueryType::TIMESTAMP)
                            .query_count(GPU_TIMESTAMP_COUNT),
                        None,
                    )
                    .unwrap()
            };
            FrameSlot {
                command_pool,
                command_buffer,
                fence,
                query_pool,
                timestamps_pending: false,
                profile: FrameProfile {
                    plan: FrameExecutionPlan::Empty,
                    geometry_upload: false,
                    postprocess: false,
                    output: false,
                },
            }
        });
        Self { ctx, slots }
    }

    pub(super) fn begin(
        &mut self,
        frame_index: usize,
        profiling: bool,
    ) -> (FrameRecording<'_>, Option<GpuPassTimings>) {
        let slot = &mut self.slots[frame_index];
        let device = &self.ctx.device;
        unsafe {
            device
                .wait_for_fences(std::slice::from_ref(&slot.fence), true, u64::MAX)
                .unwrap();
        }
        let timings = slot.timestamps_pending.then(|| {
            let mut timestamps = [0u64; GPU_TIMESTAMP_COUNT as usize];
            unsafe {
                device
                    .get_query_pool_results(
                        slot.query_pool,
                        0,
                        &mut timestamps,
                        vk::QueryResultFlags::TYPE_64,
                    )
                    .unwrap();
            }
            slot.timestamps_pending = false;
            GpuPassTimings::from_timestamps(
                timestamps,
                self.ctx.timestamp_period_ns,
                self.ctx.timestamp_valid_bits,
                slot.profile.plan,
                slot.profile.geometry_upload,
                slot.profile.postprocess,
                slot.profile.output,
            )
        });
        unsafe {
            device
                .reset_fences(std::slice::from_ref(&slot.fence))
                .unwrap();
            device
                .reset_command_pool(slot.command_pool, vk::CommandPoolResetFlags::empty())
                .unwrap();
            device
                .begin_command_buffer(
                    slot.command_buffer,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .unwrap();
            if profiling {
                device.cmd_reset_query_pool(
                    slot.command_buffer,
                    slot.query_pool,
                    0,
                    GPU_TIMESTAMP_COUNT,
                );
                write_gpu_timestamp(device, slot.command_buffer, slot.query_pool, 0, true);
            }
        }
        (
            FrameRecording {
                ctx: &self.ctx,
                slot,
            },
            timings,
        )
    }

    pub(super) fn wait(&self, frame_index: usize) {
        unsafe {
            self.ctx
                .device
                .wait_for_fences(
                    std::slice::from_ref(&self.slots[frame_index].fence),
                    true,
                    u64::MAX,
                )
                .unwrap();
        }
    }
}

impl FrameRecording<'_> {
    pub(super) fn command_buffer(&self) -> vk::CommandBuffer {
        self.slot.command_buffer
    }

    pub(super) fn query_pool(&self) -> vk::QueryPool {
        self.slot.query_pool
    }

    pub(super) unsafe fn submit(
        self,
        wait_infos: &[vk::SemaphoreSubmitInfo<'_>],
        signal_infos: &[vk::SemaphoreSubmitInfo<'_>],
        profile: Option<FrameProfile>,
    ) {
        unsafe {
            self.ctx
                .device
                .end_command_buffer(self.slot.command_buffer)
                .unwrap();
            let command_buffer_info =
                vk::CommandBufferSubmitInfo::default().command_buffer(self.slot.command_buffer);
            let submit_info = vk::SubmitInfo2::default()
                .wait_semaphore_infos(wait_infos)
                .command_buffer_infos(std::slice::from_ref(&command_buffer_info))
                .signal_semaphore_infos(signal_infos);
            self.ctx
                .device
                .queue_submit2(
                    self.ctx.queue,
                    std::slice::from_ref(&submit_info),
                    self.slot.fence,
                )
                .unwrap();
        }
        if let Some(profile) = profile {
            self.slot.timestamps_pending = true;
            self.slot.profile = profile;
        }
    }
}

impl Drop for FrameSet {
    fn drop(&mut self) {
        unsafe {
            let _ = self.ctx.device.device_wait_idle();
            for slot in &self.slots {
                self.ctx.device.destroy_query_pool(slot.query_pool, None);
                self.ctx.device.destroy_fence(slot.fence, None);
                self.ctx
                    .device
                    .destroy_command_pool(slot.command_pool, None);
            }
        }
    }
}
