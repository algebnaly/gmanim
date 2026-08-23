use std::sync::Arc;

use ash::vk;
use ash::vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_MAIN;

use crate::vulkan::context::{TimelineSemaphore, VulkanContext};

pub(super) const VIDEO_NV12_IMAGE_COUNT: usize = 9;
const VK_FORMAT_G8_B8R8_2PLANE_420_UNORM_RAW: i32 = 1_000_156_003;
const VK_IMAGE_USAGE_VIDEO_ENCODE_SRC_BIT_KHR_RAW: u32 = 0x0000_4000;
const VK_IMAGE_ASPECT_PLANE_0_BIT_RAW: u32 = 0x0000_0010;
const VK_IMAGE_ASPECT_PLANE_1_BIT_RAW: u32 = 0x0000_0020;

pub(super) fn video_timeline_values(next_ready_value: u64) -> (Option<u64>, u64, u64) {
    assert!(
        next_ready_value % 2 == 1,
        "video ready timeline values must be odd"
    );
    (
        (next_ready_value > 1).then_some(next_ready_value - 1),
        next_ready_value,
        next_ready_value + 1,
    )
}

pub(super) struct VideoNv12Image {
    ctx: Arc<VulkanContext>,
    pub vk_image: vk::Image,
    pub allocation: Option<gpu_allocator::vulkan::Allocation>,
    pub color_view: vk::ImageView,
    pub y_view: vk::ImageView,
    pub uv_view: vk::ImageView,
    pub format: vk::Format,
    pub width: u32,
    pub height: u32,
}

impl VideoNv12Image {
    pub fn new(ctx: &Arc<VulkanContext>, width: u32, height: u32) -> Self {
        let format = vk::Format::from_raw(VK_FORMAT_G8_B8R8_2PLANE_420_UNORM_RAW);
        let usage = vk::ImageUsageFlags::STORAGE
            | vk::ImageUsageFlags::from_raw(VK_IMAGE_USAGE_VIDEO_ENCODE_SRC_BIT_KHR_RAW);
        let mut queue_family_indices = Vec::new();
        queue_family_indices.push(ctx.queue_family_index);
        if let Some(video_queue_family_index) = ctx.video_encode_queue_family_index {
            if video_queue_family_index != ctx.queue_family_index {
                queue_family_indices.push(video_queue_family_index);
            }
        }
        let sharing_mode = if queue_family_indices.len() > 1 {
            vk::SharingMode::CONCURRENT
        } else {
            vk::SharingMode::EXCLUSIVE
        };
        let mut h264_profile = vk::VideoEncodeH264ProfileInfoKHR::default()
            .std_profile_idc(StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_MAIN);
        let profile = vk::VideoProfileInfoKHR::default()
            .video_codec_operation(vk::VideoCodecOperationFlagsKHR::ENCODE_H264)
            .chroma_subsampling(vk::VideoChromaSubsamplingFlagsKHR::TYPE_420)
            .luma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
            .chroma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
            .push_next(&mut h264_profile);
        let profiles = [profile];
        let mut profile_list = vk::VideoProfileListInfoKHR::default().profiles(&profiles);
        let image_info = vk::ImageCreateInfo {
            s_type: vk::StructureType::IMAGE_CREATE_INFO,
            p_next: (&mut profile_list as *mut vk::VideoProfileListInfoKHR).cast(),
            flags: vk::ImageCreateFlags::MUTABLE_FORMAT | vk::ImageCreateFlags::EXTENDED_USAGE,
            image_type: vk::ImageType::TYPE_2D,
            format,
            extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage,
            sharing_mode,
            queue_family_index_count: queue_family_indices.len() as u32,
            p_queue_family_indices: queue_family_indices.as_ptr(),
            initial_layout: vk::ImageLayout::UNDEFINED,
            ..Default::default()
        };

        let vk_image = unsafe { ctx.device.create_image(&image_info, None).unwrap() };
        let requirements = unsafe { ctx.device.get_image_memory_requirements(vk_image) };
        let allocation = ctx
            .allocator
            .lock()
            .unwrap()
            .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                name: "video-nv12-image",
                requirements,
                location: gpu_allocator::MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            })
            .unwrap();

        unsafe {
            ctx.device
                .bind_image_memory(vk_image, allocation.memory(), allocation.offset())
                .unwrap();
        }

        let color_view = Self::create_view(
            ctx,
            vk_image,
            format,
            vk::ImageAspectFlags::COLOR,
            vk::ImageUsageFlags::from_raw(VK_IMAGE_USAGE_VIDEO_ENCODE_SRC_BIT_KHR_RAW),
        );
        let y_view = Self::create_view(
            ctx,
            vk_image,
            vk::Format::R8_UNORM,
            vk::ImageAspectFlags::from_raw(VK_IMAGE_ASPECT_PLANE_0_BIT_RAW),
            vk::ImageUsageFlags::STORAGE,
        );
        let uv_view = Self::create_view(
            ctx,
            vk_image,
            vk::Format::R8G8_UNORM,
            vk::ImageAspectFlags::from_raw(VK_IMAGE_ASPECT_PLANE_1_BIT_RAW),
            vk::ImageUsageFlags::STORAGE,
        );

        Self {
            ctx: Arc::clone(ctx),
            vk_image,
            allocation: Some(allocation),
            color_view,
            y_view,
            uv_view,
            format,
            width,
            height,
        }
    }

    fn create_view(
        ctx: &VulkanContext,
        image: vk::Image,
        format: vk::Format,
        aspect_mask: vk::ImageAspectFlags,
        usage: vk::ImageUsageFlags,
    ) -> vk::ImageView {
        let mut usage_info = vk::ImageViewUsageCreateInfo::default().usage(usage);
        let view_info = vk::ImageViewCreateInfo {
            s_type: vk::StructureType::IMAGE_VIEW_CREATE_INFO,
            p_next: (&mut usage_info as *mut vk::ImageViewUsageCreateInfo).cast(),
            image,
            view_type: vk::ImageViewType::TYPE_2D,
            format,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
            ..Default::default()
        };
        unsafe { ctx.device.create_image_view(&view_info, None).unwrap() }
    }

    fn release(&mut self) {
        unsafe {
            if self.uv_view != vk::ImageView::null() {
                self.ctx.device.destroy_image_view(self.uv_view, None);
                self.uv_view = vk::ImageView::null();
            }
            if self.y_view != vk::ImageView::null() {
                self.ctx.device.destroy_image_view(self.y_view, None);
                self.y_view = vk::ImageView::null();
            }
            if self.color_view != vk::ImageView::null() {
                self.ctx.device.destroy_image_view(self.color_view, None);
                self.color_view = vk::ImageView::null();
            }
            if self.vk_image != vk::Image::null() {
                self.ctx.device.destroy_image(self.vk_image, None);
                self.vk_image = vk::Image::null();
            }
        }
        if let Some(allocation) = self.allocation.take() {
            self.ctx.allocator.lock().unwrap().free(allocation).unwrap();
        }
    }
}

impl Drop for VideoNv12Image {
    fn drop(&mut self) {
        self.release();
    }
}

pub(super) struct VideoNv12Slot {
    pub image: VideoNv12Image,
    pub descriptor_set: vk::DescriptorSet,
    pub layout: vk::ImageLayout,
    pub timeline: Arc<TimelineSemaphore>,
    pub next_ready_value: u64,
    pub last_ready_value: Option<u64>,
    pub frame_available: bool,
}

impl VideoNv12Slot {
    pub(super) fn new(ctx: &Arc<VulkanContext>, width: u32, height: u32) -> Self {
        Self {
            image: VideoNv12Image::new(ctx, width, height),
            descriptor_set: vk::DescriptorSet::null(),
            layout: vk::ImageLayout::UNDEFINED,
            timeline: TimelineSemaphore::new(ctx, 0)
                .expect("failed to create video frame timeline semaphore"),
            next_ready_value: 1,
            last_ready_value: None,
            frame_available: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::video_timeline_values;

    #[test]
    fn timeline_values_alternate_ready_and_release() {
        assert_eq!(video_timeline_values(1), (None, 1, 2));
        assert_eq!(video_timeline_values(3), (Some(2), 3, 4));
        assert_eq!(video_timeline_values(9), (Some(8), 9, 10));
    }

    #[test]
    #[should_panic(expected = "must be odd")]
    fn timeline_rejects_even_ready_values() {
        video_timeline_values(2);
    }
}
