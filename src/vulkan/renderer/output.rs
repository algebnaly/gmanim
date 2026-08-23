#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RenderOutputs {
    pub cpu_nv12: bool,
    pub vulkan_video: bool,
    pub cpu_rgba: bool,
    pub cpu_yuv444p: bool,
}

impl RenderOutputs {
    pub const ALL: Self = Self {
        cpu_nv12: true,
        vulkan_video: true,
        cpu_rgba: true,
        cpu_yuv444p: true,
    };

    pub const VULKAN_VIDEO_ONLY: Self = Self {
        cpu_nv12: false,
        vulkan_video: true,
        cpu_rgba: false,
        cpu_yuv444p: false,
    };

    pub const CPU_NV12_ONLY: Self = Self {
        cpu_nv12: true,
        vulkan_video: false,
        cpu_rgba: false,
        cpu_yuv444p: false,
    };

    pub const CPU_RGBA_ONLY: Self = Self {
        cpu_nv12: false,
        vulkan_video: false,
        cpu_rgba: true,
        cpu_yuv444p: false,
    };

    pub const CPU_READBACKS: Self = Self {
        cpu_nv12: true,
        vulkan_video: false,
        cpu_rgba: true,
        cpu_yuv444p: true,
    };

    pub const NONE: Self = Self {
        cpu_nv12: false,
        vulkan_video: false,
        cpu_rgba: false,
        cpu_yuv444p: false,
    };
}

pub(super) struct OutputRetirement {
    ctx: Arc<VulkanContext>,
    rgba_readback: Vec<u8>,
}

pub(super) struct OutputSubmission {
    frame_index: usize,
    video_frame_index: usize,
    video_ready_value: Option<u64>,
    wait_infos: [vk::SemaphoreSubmitInfo<'static>; 1],
    wait_count: usize,
    signal_infos: [vk::SemaphoreSubmitInfo<'static>; 1],
    signal_count: usize,
}

impl OutputRetirement {
    pub(super) fn new(ctx: Arc<VulkanContext>) -> Self {
        Self {
            ctx,
            rgba_readback: Vec::new(),
        }
    }

    pub(super) fn prepare_submission(
        &self,
        cache: &TargetCache,
        plan: RecordingPlan,
        outputs: RenderOutputs,
    ) -> OutputSubmission {
        let frame_index = cache.current_frame % RENDER_FRAME_COUNT;
        let video_frame_index = cache.current_frame % cache.video_nv12_slots.len();
        let mut wait_infos = [vk::SemaphoreSubmitInfo::default()];
        let mut signal_infos = [vk::SemaphoreSubmitInfo::default()];
        let mut wait_count = 0;
        let mut signal_count = 0;
        let mut video_ready_value = None;

        if outputs.vulkan_video {
            if cache.current_frame > 0 {
                let previous_video_frame_index =
                    (cache.current_frame - 1) % cache.video_nv12_slots.len();
                assert!(
                    !cache.video_nv12_slots[previous_video_frame_index].frame_available,
                    "the previous Vulkan video frame must be acquired before rendering another frame"
                );
            }

            let slot = &cache.video_nv12_slots[video_frame_index];
            let input_info = vk::DescriptorImageInfo {
                image_view: if plan.fused_video_downsample {
                    cache.render_targets[frame_index].resolved_texture.view
                } else {
                    cache.render_targets[frame_index].texture.view
                },
                image_layout: vk::ImageLayout::GENERAL,
                ..Default::default()
            };
            let input_write = vk::WriteDescriptorSet {
                s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                dst_set: slot.descriptor_set,
                dst_binding: 0,
                descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                descriptor_count: 1,
                p_image_info: &input_info,
                ..Default::default()
            };
            unsafe {
                self.ctx
                    .device
                    .update_descriptor_sets(std::slice::from_ref(&input_write), &[]);
            }

            let (wait_value, ready_value, _) = video_timeline_values(slot.next_ready_value);
            if let Some(wait_value) = wait_value {
                wait_infos[0] = vk::SemaphoreSubmitInfo::default()
                    .semaphore(slot.timeline.handle())
                    .value(wait_value)
                    .stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER);
                wait_count = 1;
            }
            signal_infos[0] = vk::SemaphoreSubmitInfo::default()
                .semaphore(slot.timeline.handle())
                .value(ready_value)
                .stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER);
            signal_count = 1;
            video_ready_value = Some(ready_value);
        }

        OutputSubmission {
            frame_index,
            video_frame_index,
            video_ready_value,
            wait_infos,
            wait_count,
            signal_infos,
            signal_count,
        }
    }

    pub(super) fn copy_submitted_rgba(
        &self,
        cache: &TargetCache,
        frames: &FrameSet,
        frame_index: usize,
        output: &mut [u8],
    ) {
        frames.wait(frame_index);
        Self::copy_rgba_rows(cache, frame_index, output)
            .expect("submitted RGBA readback buffer is not mapped");
    }

    pub(super) fn current_image_view(&self, cache: &TargetCache) -> Option<vk::ImageView> {
        let frame_index = Self::latest_frame_index(cache)?;
        Some(cache.render_targets[frame_index].texture.view)
    }

    pub(super) fn latest_nv12<'a>(
        &self,
        cache: &'a TargetCache,
        frames: &FrameSet,
    ) -> Option<&'a [u8]> {
        let frame_index = Self::latest_frame_index(cache)?;
        frames.wait(frame_index);
        let len = (cache.width * cache.height * 3 / 2) as usize;
        Self::mapped_bytes(&cache.nv12_output_buffers[frame_index], len)
    }

    pub(super) fn latest_yuv444p<'a>(
        &self,
        cache: &'a TargetCache,
        frames: &FrameSet,
    ) -> Option<&'a [u8]> {
        let frame_index = Self::latest_frame_index(cache)?;
        frames.wait(frame_index);
        let len = (cache.width * cache.height * 3) as usize;
        Self::mapped_bytes(&cache.yuv444p_output_buffers[frame_index], len)
    }

    pub(super) fn take_video_frame(&self, cache: &mut TargetCache) -> Option<VulkanVideoFrame> {
        if cache.current_frame == 0 || cache.video_nv12_slots.is_empty() {
            return None;
        }
        let frame_index = (cache.current_frame - 1) % cache.video_nv12_slots.len();
        let slot = &mut cache.video_nv12_slots[frame_index];
        if !slot.frame_available {
            return None;
        }
        let ready_value = slot.last_ready_value?;
        let (_, _, release_value) = video_timeline_values(ready_value);
        slot.frame_available = false;
        Some(VulkanVideoFrame::new(
            slot.image.vk_image,
            slot.image.color_view,
            slot.layout,
            slot.image.format,
            slot.image.width,
            slot.image.height,
            self.ctx.device.handle(),
            slot.timeline.clone(),
            ready_value,
            release_value,
        ))
    }

    pub(super) fn latest_rgba<'a>(
        &'a mut self,
        cache: &TargetCache,
        frames: &FrameSet,
    ) -> Option<&'a [u8]> {
        let frame_index = Self::latest_frame_index(cache)?;
        frames.wait(frame_index);
        self.rgba_readback
            .resize((cache.width * cache.height * 4) as usize, 0);
        Self::copy_rgba_rows(cache, frame_index, &mut self.rgba_readback)?;
        Some(&self.rgba_readback)
    }

    fn latest_frame_index(cache: &TargetCache) -> Option<usize> {
        (cache.current_frame > 0).then(|| (cache.current_frame - 1) % RENDER_FRAME_COUNT)
    }

    fn mapped_bytes(buffer: &super::Buffer, len: usize) -> Option<&[u8]> {
        let mapped = buffer.allocation.as_ref()?.mapped_ptr()?;
        Some(unsafe { std::slice::from_raw_parts(mapped.as_ptr().cast::<u8>(), len) })
    }

    fn copy_rgba_rows(cache: &TargetCache, frame_index: usize, output: &mut [u8]) -> Option<()> {
        let row_bytes = cache.width * 4;
        let required_len = (row_bytes * cache.height) as usize;
        assert!(
            output.len() >= required_len,
            "RGBA output buffer requires {required_len} bytes, got {}",
            output.len()
        );
        let padded = Self::mapped_bytes(
            &cache.output_buffers[frame_index],
            (cache.padded_bytes_per_row * cache.height) as usize,
        )?;
        for row in 0..cache.height {
            let src_start = (row * cache.padded_bytes_per_row) as usize;
            let src_end = src_start + row_bytes as usize;
            let dst_start = (row * row_bytes) as usize;
            let dst_end = dst_start + row_bytes as usize;
            output[dst_start..dst_end].copy_from_slice(&padded[src_start..src_end]);
        }
        Some(())
    }
}

impl OutputSubmission {
    pub(super) fn frame_index(&self) -> usize {
        self.frame_index
    }

    pub(super) fn video_frame_index(&self) -> usize {
        self.video_frame_index
    }

    pub(super) fn wait_infos(&self) -> &[vk::SemaphoreSubmitInfo<'_>] {
        &self.wait_infos[..self.wait_count]
    }

    pub(super) fn signal_infos(&self) -> &[vk::SemaphoreSubmitInfo<'_>] {
        &self.signal_infos[..self.signal_count]
    }

    pub(super) fn commit(self, cache: &mut TargetCache) {
        if let Some(ready_value) = self.video_ready_value {
            let slot = &mut cache.video_nv12_slots[self.video_frame_index];
            debug_assert_eq!(slot.next_ready_value, ready_value);
            slot.last_ready_value = Some(ready_value);
            slot.next_ready_value += 2;
            slot.frame_available = true;
        }
        cache.current_frame += 1;
    }
}
use std::sync::Arc;

use ash::vk;

use crate::video_backend::vulkan_h264::VulkanVideoFrame;
use crate::vulkan::context::VulkanContext;

use super::frame::{FrameSet, RENDER_FRAME_COUNT};
use super::record::RecordingPlan;
use super::targets::TargetCache;
use super::video_output::video_timeline_values;
