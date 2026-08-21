use ash::vk;

use crate::vulkan::context::VulkanContext;

pub struct Buffer {
    pub vk_buffer: vk::Buffer,
    pub allocation: Option<gpu_allocator::vulkan::Allocation>,
    pub size: u64,
}

impl Buffer {
    pub fn new(
        ctx: &VulkanContext,
        size: u64,
        usage: vk::BufferUsageFlags,
        memory_location: gpu_allocator::MemoryLocation,
    ) -> Self {
        let buffer_info = vk::BufferCreateInfo {
            s_type: vk::StructureType::BUFFER_CREATE_INFO,
            size,
            usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        let vk_buffer = unsafe { ctx.device.create_buffer(&buffer_info, None).unwrap() };
        let requirements = unsafe { ctx.device.get_buffer_memory_requirements(vk_buffer) };

        let allocation = ctx
            .allocator
            .lock()
            .unwrap()
            .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                name: "buffer",
                requirements,
                location: memory_location,
                linear: true,
                allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            })
            .unwrap();

        unsafe {
            ctx.device
                .bind_buffer_memory(vk_buffer, allocation.memory(), allocation.offset())
                .unwrap();
        }

        Self {
            vk_buffer,
            allocation: Some(allocation),
            size,
        }
    }

    pub fn write_bytes(&self, offset: u64, data: &[u8]) {
        if let Some(mapped) = self
            .allocation
            .as_ref()
            .and_then(gpu_allocator::vulkan::Allocation::mapped_ptr)
        {
            unsafe {
                let dst = mapped.as_ptr().add(offset as usize);
                std::ptr::copy_nonoverlapping(data.as_ptr(), dst.cast::<u8>(), data.len());
            }
        }
    }

    pub fn destroy(&mut self, ctx: &VulkanContext) {
        unsafe {
            ctx.device.destroy_buffer(self.vk_buffer, None);
        }
        if let Some(allocation) = self.allocation.take() {
            ctx.allocator.lock().unwrap().free(allocation).unwrap();
        }
    }
}

pub struct Image {
    pub vk_image: vk::Image,
    pub allocation: Option<gpu_allocator::vulkan::Allocation>,
    pub view: vk::ImageView,
    pub format: vk::Format,
    pub width: u32,
    pub height: u32,
    pub mip_levels: u32,
}

impl Image {
    pub fn new(
        ctx: &VulkanContext,
        width: u32,
        height: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        aspect_mask: vk::ImageAspectFlags,
        samples: vk::SampleCountFlags,
    ) -> Self {
        Self::new_with_mip_levels(ctx, width, height, format, usage, aspect_mask, samples, 1)
    }

    pub fn new_with_mip_levels(
        ctx: &VulkanContext,
        width: u32,
        height: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        aspect_mask: vk::ImageAspectFlags,
        samples: vk::SampleCountFlags,
        mip_levels: u32,
    ) -> Self {
        let image_info = vk::ImageCreateInfo {
            s_type: vk::StructureType::IMAGE_CREATE_INFO,
            image_type: vk::ImageType::TYPE_2D,
            format,
            extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
            mip_levels,
            array_layers: 1,
            samples,
            tiling: vk::ImageTiling::OPTIMAL,
            usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
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
                name: "image",
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

        let view_info = vk::ImageViewCreateInfo {
            s_type: vk::StructureType::IMAGE_VIEW_CREATE_INFO,
            image: vk_image,
            view_type: vk::ImageViewType::TYPE_2D,
            format,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask,
                base_mip_level: 0,
                level_count: mip_levels,
                base_array_layer: 0,
                layer_count: 1,
            },
            ..Default::default()
        };

        let view = unsafe { ctx.device.create_image_view(&view_info, None).unwrap() };

        Self {
            vk_image,
            allocation: Some(allocation),
            view,
            format,
            width,
            height,
            mip_levels,
        }
    }

    pub fn destroy(&mut self, ctx: &VulkanContext) {
        unsafe {
            ctx.device.destroy_image_view(self.view, None);
            ctx.device.destroy_image(self.vk_image, None);
        }
        if let Some(allocation) = self.allocation.take() {
            ctx.allocator.lock().unwrap().free(allocation).unwrap();
        }
    }
}
