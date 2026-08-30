use ash::vk;
use std::sync::Arc;

use crate::vulkan::context::VulkanContext;

use super::Buffer;
use super::{DescriptorPool, Image, PipelineSet};

fn normalized(direction: [f32; 3]) -> [f32; 3] {
    let inverse_length = 1.0
        / (direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2])
            .sqrt();
    [
        direction[0] * inverse_length,
        direction[1] * inverse_length,
        direction[2] * inverse_length,
    ]
}

fn studio_environment(direction: [f32; 3], roughness: f32) -> [f32; 4] {
    let key = normalized([-0.55, 0.65, 0.52]);
    let rim = normalized([0.72, 0.2, -0.66]);
    let ceiling = normalized([0.0, 1.0, 0.08]);
    let dot = |axis: [f32; 3]| {
        (direction[0] * axis[0] + direction[1] * axis[1] + direction[2] * axis[2]).max(0.0)
    };
    let sharpness = 2.0 + 126.0 * (1.0 - roughness).powi(4);
    let key_light = dot(key).powf(sharpness) * (7.5 - 5.5 * roughness);
    let rim_light = dot(rim).powf(sharpness * 0.55) * (3.2 - 2.0 * roughness);
    let ceiling_light = dot(ceiling).powf(4.0 + sharpness * 0.12) * 1.4;
    let horizon = (1.0 - direction[1].abs()).powf(3.0) * 0.18;
    let ambient = 0.04 + horizon * 0.9;
    [
        ambient + key_light + rim_light * 0.65 + ceiling_light * 0.7,
        ambient + key_light * 0.96 + rim_light * 0.65 + ceiling_light * 0.7,
        ambient + key_light * 0.92 + rim_light * 0.65 + ceiling_light * 0.7,
        1.0,
    ]
}

fn create_studio_environment(ctx: &Arc<VulkanContext>) -> (Image, vk::Sampler) {
    const WIDTH: u32 = 256;
    const HEIGHT: u32 = 128;
    const MIP_LEVELS: u32 = 9;

    let mut pixels = Vec::<f32>::new();
    let mut regions = Vec::with_capacity(MIP_LEVELS as usize);
    let mut byte_offset = 0_u64;
    for mip_level in 0..MIP_LEVELS {
        let width = (WIDTH >> mip_level).max(1);
        let height = (HEIGHT >> mip_level).max(1);
        let roughness = mip_level as f32 / (MIP_LEVELS - 1) as f32;
        regions.push(
            vk::BufferImageCopy::default()
                .buffer_offset(byte_offset)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                }),
        );
        for y in 0..height {
            let theta = (y as f32 + 0.5) / height as f32 * std::f32::consts::PI;
            let sin_theta = theta.sin();
            for x in 0..width {
                let phi = ((x as f32 + 0.5) / width as f32 - 0.5) * 2.0 * std::f32::consts::PI;
                pixels.extend(studio_environment(
                    [sin_theta * phi.cos(), theta.cos(), sin_theta * phi.sin()],
                    roughness,
                ));
            }
        }
        byte_offset += (width * height * 4 * std::mem::size_of::<f32>() as u32) as u64;
    }

    let staging = Buffer::new(
        ctx,
        byte_offset,
        vk::BufferUsageFlags::TRANSFER_SRC,
        gpu_allocator::MemoryLocation::CpuToGpu,
    );
    staging.write_bytes(0, bytemuck::cast_slice(&pixels));
    let image = Image::new_with_mip_levels(
        ctx,
        vk::Extent2D {
            width: WIDTH,
            height: HEIGHT,
        },
        vk::Format::R32G32B32A32_SFLOAT,
        vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
        vk::ImageAspectFlags::COLOR,
        vk::SampleCountFlags::TYPE_1,
        MIP_LEVELS,
    );

    submit_immediate(ctx, |command_buffer| unsafe {
        let upload_barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::COPY)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .image(image.vk_image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: MIP_LEVELS,
                base_array_layer: 0,
                layer_count: 1,
            });
        ctx.device.cmd_pipeline_barrier2(
            command_buffer,
            &vk::DependencyInfo::default()
                .image_memory_barriers(std::slice::from_ref(&upload_barrier)),
        );
        ctx.device.cmd_copy_buffer_to_image(
            command_buffer,
            staging.vk_buffer,
            image.vk_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &regions,
        );
        let sample_barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COPY)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
            .dst_access_mask(vk::AccessFlags2::SHADER_READ)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image(image.vk_image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: MIP_LEVELS,
                base_array_layer: 0,
                layer_count: 1,
            });
        ctx.device.cmd_pipeline_barrier2(
            command_buffer,
            &vk::DependencyInfo::default()
                .image_memory_barriers(std::slice::from_ref(&sample_barrier)),
        );
    });

    let sampler = unsafe {
        ctx.device
            .create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::REPEAT)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .min_lod(0.0)
                    .max_lod((MIP_LEVELS - 1) as f32),
                None,
            )
            .unwrap()
    };
    (image, sampler)
}

pub(super) struct StudioEnvironment {
    device: Arc<ash::Device>,
    pub(super) image: Image,
    pub(super) sampler: vk::Sampler,
}

impl StudioEnvironment {
    pub(super) fn new(ctx: &Arc<VulkanContext>) -> Self {
        let (image, sampler) = create_studio_environment(ctx);
        Self {
            device: Arc::clone(&ctx.device),
            image,
            sampler,
        }
    }
}

impl Drop for StudioEnvironment {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_sampler(self.sampler, None);
        }
    }
}

pub(super) struct TextureRegistry {
    ctx: Arc<VulkanContext>,
    descriptor_pool: Arc<DescriptorPool>,
    descriptor_set: vk::DescriptorSet,
    sampler: vk::Sampler,
    images: Vec<Image>,
}

impl TextureRegistry {
    pub(super) fn new(
        ctx: &Arc<VulkanContext>,
        descriptor_pool: &Arc<DescriptorPool>,
        pipelines: &PipelineSet,
    ) -> Self {
        let sampler = unsafe {
            ctx.device
                .create_sampler(
                    &vk::SamplerCreateInfo::default()
                        .mag_filter(vk::Filter::LINEAR)
                        .min_filter(vk::Filter::LINEAR)
                        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                    None,
                )
                .unwrap()
        };
        let descriptor_set = unsafe {
            ctx.device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(descriptor_pool.handle())
                        .set_layouts(std::slice::from_ref(&pipelines.raster_texture_layout)),
                )
                .unwrap()[0]
        };
        let dummy = Image::new(
            ctx,
            1,
            1,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            vk::ImageAspectFlags::COLOR,
            vk::SampleCountFlags::TYPE_1,
        );
        submit_immediate(ctx, |command_buffer| unsafe {
            let barrier = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(dummy.vk_image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            ctx.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                std::slice::from_ref(&barrier),
            );
        });

        let dummy_infos = vec![
            vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(dummy.view);
            16
        ];
        let sampler_infos = [vk::DescriptorImageInfo::default().sampler(sampler)];
        unsafe {
            ctx.device.update_descriptor_sets(
                &[
                    vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_set)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                        .image_info(&dummy_infos),
                    vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_set)
                        .dst_binding(1)
                        .descriptor_type(vk::DescriptorType::SAMPLER)
                        .image_info(&sampler_infos),
                ],
                &[],
            );
        }

        Self {
            ctx: Arc::clone(ctx),
            descriptor_pool: Arc::clone(descriptor_pool),
            descriptor_set,
            sampler,
            images: vec![dummy],
        }
    }

    pub(super) fn descriptor_set(&self) -> vk::DescriptorSet {
        self.descriptor_set
    }

    pub(super) fn update(&mut self, index: u32, width: u32, height: u32, data: &[u8]) {
        assert!(index < 16, "Texture index out of bounds");
        let staging = Buffer::new(
            &self.ctx,
            (width * height * 4) as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            gpu_allocator::MemoryLocation::CpuToGpu,
        );
        staging.write_bytes(0, data);
        let image = Image::new(
            &self.ctx,
            width,
            height,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
            vk::ImageAspectFlags::COLOR,
            vk::SampleCountFlags::TYPE_1,
        );
        submit_immediate(&self.ctx, |command_buffer| unsafe {
            let range = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            };
            let upload_barrier = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image.vk_image)
                .subresource_range(range)
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
            self.ctx.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                std::slice::from_ref(&upload_barrier),
            );
            self.ctx.device.cmd_copy_buffer_to_image(
                command_buffer,
                staging.vk_buffer,
                image.vk_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::BufferImageCopy::default()
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image_extent(vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    })],
            );
            let sample_barrier = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image.vk_image)
                .subresource_range(range)
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            self.ctx.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                std::slice::from_ref(&sample_barrier),
            );
        });
        self.bind(index, image.view);
        self.images.push(image);
    }

    pub(super) fn bind(&self, index: u32, view: vk::ImageView) {
        assert!(index < 16, "Texture index out of bounds");
        let image_infos = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(view)];
        unsafe {
            self.ctx.device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(self.descriptor_set)
                    .dst_binding(0)
                    .dst_array_element(index)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(&image_infos)],
                &[],
            );
        }
    }
}

impl Drop for TextureRegistry {
    fn drop(&mut self) {
        self.descriptor_pool
            .free(std::slice::from_ref(&self.descriptor_set));
        unsafe {
            self.ctx.device.destroy_sampler(self.sampler, None);
        }
    }
}

fn submit_immediate(ctx: &VulkanContext, record: impl FnOnce(vk::CommandBuffer)) {
    let command_pool = unsafe {
        ctx.device
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(ctx.queue_family_index)
                    .flags(vk::CommandPoolCreateFlags::TRANSIENT),
                None,
            )
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
    unsafe {
        ctx.device
            .begin_command_buffer(
                command_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .unwrap();
    }
    record(command_buffer);
    unsafe {
        ctx.device.end_command_buffer(command_buffer).unwrap();
        ctx.device
            .queue_submit(
                ctx.queue,
                &[
                    vk::SubmitInfo::default()
                        .command_buffers(std::slice::from_ref(&command_buffer)),
                ],
                vk::Fence::null(),
            )
            .unwrap();
        ctx.device.queue_wait_idle(ctx.queue).unwrap();
        ctx.device.destroy_command_pool(command_pool, None);
    }
}
