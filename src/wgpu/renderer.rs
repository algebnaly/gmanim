use crate::wgpu::context::WgpuContext;
use std::sync::Arc;

pub struct RenderCache {
    pub width: u32,
    pub height: u32,
    pub texture: wgpu::Texture,
    pub output_buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
    pub padded_bytes_per_row: u32,
}

pub struct WgpuRenderer {
    ctx: Arc<WgpuContext>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    camera_buffer: wgpu::Buffer,
    buffer_3d: wgpu::Buffer,
    cache: std::sync::Mutex<Option<RenderCache>>,
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
    pub _padding2: [u32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PrimitiveData3D {
    pub color: [f32; 4],
    pub params: [f32; 12],
    pub shape_type: u32,
    pub padding: [u32; 3],
}

impl WgpuRenderer {
    pub fn new(ctx: Arc<WgpuContext>) -> Self {
        let shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("3D Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
            });

        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("3D Bind Group Layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::StorageTexture {
                                access: wgpu::StorageTextureAccess::WriteOnly,
                                format: wgpu::TextureFormat::Rgba8Unorm,
                                view_dimension: wgpu::TextureViewDimension::D2,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("3D Pipeline Layout"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });

        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("3D Compute Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let camera_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Camera Buffer"),
            size: std::mem::size_of::<CameraUniform>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // initial capacity for 100 primitives
        let buffer_3d = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("3D Buffer"),
            size: (std::mem::size_of::<PrimitiveData3D>() * 100) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            ctx,
            pipeline,
            bind_group_layout,
            camera_buffer,
            buffer_3d,
            cache: std::sync::Mutex::new(None),
        }
    }

    pub fn render(
        &self,
        width: u32,
        height: u32,
        camera_uniform: &CameraUniform,
        objects_3d: &[PrimitiveData3D],
        output: &mut [u8],
    ) {
        self.ctx
            .queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(camera_uniform));
        if !objects_3d.is_empty() {
            let bytes_3d = bytemuck::cast_slice(objects_3d);
            // Reallocate if too small
            if bytes_3d.len() as u64 > self.buffer_3d.size() {
                // Ignore reallocation for simplicity in this prototype, just slice it
                let len = (self.buffer_3d.size() as usize).min(bytes_3d.len());
                self.ctx
                    .queue
                    .write_buffer(&self.buffer_3d, 0, &bytes_3d[..len]);
            } else {
                self.ctx.queue.write_buffer(&self.buffer_3d, 0, bytes_3d);
            }
        }
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let unpadded_bytes_per_row = width * 4;
        let padded_bytes_per_row = (unpadded_bytes_per_row + align - 1) & !(align - 1);

        let mut cache_guard = self.cache.lock().unwrap();
        let cache_needs_update = cache_guard.as_ref().map_or(true, |c| c.width != width || c.height != height);

        if cache_needs_update {
            let texture_desc = wgpu::TextureDescriptor {
                label: Some("Render Texture"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            };
            let texture = self.ctx.device.create_texture(&texture_desc);
            let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

            let bind_group = self
                .ctx
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("3D Bind Group"),
                    layout: &self.bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&texture_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: self.camera_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self.buffer_3d.as_entire_binding(),
                        },
                    ],
                });

            let output_buffer_size = (padded_bytes_per_row * height) as wgpu::BufferAddress;
            let output_buffer = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Output Buffer"),
                size: output_buffer_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            *cache_guard = Some(RenderCache {
                width,
                height,
                texture,
                output_buffer,
                bind_group,
                padded_bytes_per_row,
            });
        }

        let cache = cache_guard.as_ref().unwrap();

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("3D Compute Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &cache.bind_group, &[]);
            let workgroup_x = (width + 15) / 16;
            let workgroup_y = (height + 15) / 16;
            cpass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
        }

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &cache.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &cache.output_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(cache.padded_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        let submission_index = self.ctx.queue.submit(Some(encoder.finish()));

        let buffer_slice = cache.output_buffer.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.ctx.device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission_index),
            timeout: None,
        });

        let padded_data = buffer_slice.get_mapped_range();
        for row in 0..height {
            let start = (row * cache.padded_bytes_per_row) as usize;
            let end = start + unpadded_bytes_per_row as usize;
            let dst_start = (row * unpadded_bytes_per_row) as usize;
            let dst_end = dst_start + unpadded_bytes_per_row as usize;
            output[dst_start..dst_end].copy_from_slice(&padded_data[start..end]);
        }

        drop(padded_data);
        cache.output_buffer.unmap();
    }
}
