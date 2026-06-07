use crate::wgpu::context::WgpuContext;
use std::sync::Arc;
use crate::mobjects::mesh_3d::Vertex;

pub struct RenderCache {
    pub width: u32,
    pub height: u32,
    pub texture: wgpu::Texture,
    pub depth_texture_view: wgpu::TextureView,
    pub output_buffers: [wgpu::Buffer; 3],
    pub current_frame: usize,
    pub bind_group: wgpu::BindGroup,
    pub raster_bind_group: wgpu::BindGroup,
    pub padded_bytes_per_row: u32,
}


impl Vertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 3] = wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x3,
        2 => Float32x4
    ];
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

pub struct WgpuRenderer {
    ctx: Arc<WgpuContext>,
    pipeline: wgpu::ComputePipeline,
    raster_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    bind_group_layout: wgpu::BindGroupLayout,
    raster_bind_group_layout: wgpu::BindGroupLayout,
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
    pub aa_level: u32,
    pub _padding2: [u32; 3],
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

        
        let raster_shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Raster Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("raster_shader.wgsl").into()),
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
                            visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
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

        let raster_bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Raster Bind Group Layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
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

        let raster_pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Raster Pipeline Layout"),
                bind_group_layouts: &[Some(&raster_bind_group_layout)],
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

        
        let raster_pipeline = ctx.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Raster Pipeline"),
            layout: Some(&raster_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &raster_shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &raster_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // No culling for now
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });

        let vertex_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vertex Buffer"),
            size: (std::mem::size_of::<Vertex>() * 10000) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let index_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Index Buffer"),
            size: (std::mem::size_of::<u32>() * 30000) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
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
            raster_bind_group_layout,
            camera_buffer,
            buffer_3d,
            raster_pipeline,
            vertex_buffer,
            index_buffer,
            cache: std::sync::Mutex::new(None),
        }
    }
    pub fn render_scene(
        &mut self,
        scene: &crate::Scene,
        scene_config: &crate::SceneConfig,
        output: Option<&mut [u8]>,
    ) {
        let output_w = scene_config.output_width as f32;
        let output_h = scene_config.output_height as f32;

        let (has_clip, clip_x, clip_y, clip_w, clip_h) = match scene.clip_rect {
            Some(crate::ClipRect::Pixel(x, y, w, h)) => {
                (true, x as f32, y as f32, w as f32, h as f32)
            },
            Some(crate::ClipRect::Logical(cx, cy, w, h)) => {
                let (o_left, o_right, o_bottom, o_top) = scene.camera.ortho_params();
                let log_w = o_right - o_left;
                let log_h = o_top - o_bottom;
                
                let tl_x = cx - w / 2.0;
                let tl_y = cy + h / 2.0;
                
                let norm_x = (tl_x - o_left) / log_w;
                let norm_y = (o_top - tl_y) / log_h;
                let norm_w = w / log_w;
                let norm_h = h / log_h;
                
                (true, norm_x * output_w, norm_y * output_h, norm_w * output_w, norm_h * output_h)
            },
            None => (false, 0.0, 0.0, 0.0, 0.0),
        };

        let mut primitives_3d = Vec::new();
        let mut mesh_vertices = Vec::new();
        let mut mesh_indices = Vec::new();
        
        fn collect_3d(m: &dyn crate::mobjects::Mobject, parent_mat: nalgebra::Matrix4<crate::GMFloat>, primitives_3d: &mut Vec<PrimitiveData3D>, mesh_vertices: &mut Vec<Vertex>, mesh_indices: &mut Vec<u32>) {
            let global_mat = parent_mat * m.get_model_matrix();
            
            if let Some(node) = m.as_scene_node() {
                if let Some(comp) = &node.component {
                    collect_3d(comp.as_ref(), global_mat, primitives_3d, mesh_vertices, mesh_indices);
                }
                for child in &node.children {
                    collect_3d(child.borrow().as_ref(), global_mat, primitives_3d, mesh_vertices, mesh_indices);
                }
            } else {
                if let Some(obj_3d) = m.as_3d() {
                    primitives_3d.push(obj_3d.as_primitive_data(global_mat));
                }
                if let Some(mesh) = m.as_mesh_3d() {
                    let base_index = mesh_vertices.len() as u32;
                    for v in &mesh.vertices {
                        let pos = nalgebra::Point3::new(v.position[0], v.position[1], v.position[2]);
                        let t_pos = global_mat.transform_point(&pos);
                        let n = nalgebra::Vector3::new(v.normal[0], v.normal[1], v.normal[2]);
                        let t_n = global_mat.transform_vector(&n).normalize();
                        mesh_vertices.push(Vertex {
                            position: [t_pos.x, t_pos.y, t_pos.z],
                            normal: [t_n.x, t_n.y, t_n.z],
                            color: v.color,
                        });
                    }
                    for i in &mesh.indices {
                        mesh_indices.push(*i + base_index);
                    }
                }
            }
        }
        
        for m in &scene.mobjects {
            collect_3d(m.borrow().as_ref(), nalgebra::Matrix4::identity(), &mut primitives_3d, &mut mesh_vertices, &mut mesh_indices);
        }

        let look = scene.camera.look_at_dir();
        let camera_uniform = CameraUniform {
            pos: [
                scene.camera.position.x as f32,
                scene.camera.position.y as f32,
                scene.camera.position.z as f32,
            ],
            _padding0: 0,
            look_at: [
                scene.camera.position.x as f32 + look.x as f32,
                scene.camera.position.y as f32 + look.y as f32,
                scene.camera.position.z as f32 + look.z as f32,
            ],
            _padding1: 0,
            up: [
                scene.camera.up_dir().x as f32,
                scene.camera.up_dir().y as f32,
                scene.camera.up_dir().z as f32,
            ],
            fov: scene.camera.fov(),
            width: output_w,
            height: output_h,
            proj_type: scene.camera.proj_type(),
            ortho_left: scene.camera.ortho_params().0,
            ortho_right: scene.camera.ortho_params().1,
            ortho_bottom: scene.camera.ortho_params().2,
            ortho_top: scene.camera.ortho_params().3,
            has_clip: if has_clip { 1 } else { 0 },
            clip_x,
            clip_y,
            clip_w,
            clip_h,
            aa_level: scene.aa_level,
            _padding2: [0; 3],
        };

        self.render(
            scene_config.output_width,
            scene_config.output_height,
            &camera_uniform,
            &primitives_3d,
            &mesh_vertices,
            &mesh_indices,
            output,
        );
    }
    pub fn render(
        &self,
        width: u32,
        height: u32,
        camera_uniform: &CameraUniform,
        objects_3d: &[PrimitiveData3D],
        mesh_vertices: &[Vertex],
        mesh_indices: &[u32],
        output: Option<&mut [u8]>,
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
        
        if !mesh_vertices.is_empty() {
            let bytes_v = bytemuck::cast_slice(mesh_vertices);
            if bytes_v.len() as u64 > self.vertex_buffer.size() {
                let len = (self.vertex_buffer.size() as usize).min(bytes_v.len());
                self.ctx.queue.write_buffer(&self.vertex_buffer, 0, &bytes_v[..len]);
            } else {
                self.ctx.queue.write_buffer(&self.vertex_buffer, 0, bytes_v);
            }
        }
        if !mesh_indices.is_empty() {
            let bytes_i = bytemuck::cast_slice(mesh_indices);
            if bytes_i.len() as u64 > self.index_buffer.size() {
                let len = (self.index_buffer.size() as usize).min(bytes_i.len());
                self.ctx.queue.write_buffer(&self.index_buffer, 0, &bytes_i[..len]);
            } else {
                self.ctx.queue.write_buffer(&self.index_buffer, 0, bytes_i);
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
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            };
            let texture = self.ctx.device.create_texture(&texture_desc);
            let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

            
            let depth_texture_desc = wgpu::TextureDescriptor {
                label: Some("Depth Texture"),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            };
            let depth_texture = self.ctx.device.create_texture(&depth_texture_desc);
            let depth_texture_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

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
            let output_buffer_0 = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Output Buffer 0"),
                size: output_buffer_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let output_buffer_1 = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Output Buffer 1"),
                size: output_buffer_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let output_buffer_2 = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Output Buffer 2"),
                size: output_buffer_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let raster_bind_group = self
                .ctx
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Raster Bind Group"),
                    layout: &self.raster_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: self.camera_buffer.as_entire_binding(),
                        },
                    ],
                });

            *cache_guard = Some(RenderCache {
                width,
                height,
                texture,
                depth_texture_view,
                output_buffers: [output_buffer_0, output_buffer_1, output_buffer_2],
                current_frame: 0,
                bind_group,
                raster_bind_group,
                padded_bytes_per_row,
            });
        }

        let cache = cache_guard.as_mut().unwrap();

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

        self.ctx.queue.submit(std::iter::once(encoder.finish()));

        let mut encoder2 = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder 2"),
            });

        let texture_view_for_render = cache.texture.create_view(&wgpu::TextureViewDescriptor::default());

        if !mesh_indices.is_empty() {
            let mut rpass = encoder2.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Mesh Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &texture_view_for_render,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load, // We load the compute pass's result!
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &cache.depth_texture_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rpass.set_pipeline(&self.raster_pipeline);
            rpass.set_bind_group(0, &cache.raster_bind_group, &[]);
            rpass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            rpass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            
            // To prevent crashing on buffer overflow we only draw up to the buffer's max indices
            let max_indices = (self.index_buffer.size() / 4) as u32;
            let indices_to_draw = (mesh_indices.len() as u32).min(max_indices);
            rpass.draw_indexed(0..indices_to_draw, 0, 0..1);
        }

        let current_buf_index = cache.current_frame % 3;
        let read_buf_index = (cache.current_frame + 1) % 3; // The buffer written 2 frames ago
        
        if let Some(_) = output {
            encoder2.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &cache.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &cache.output_buffers[current_buf_index],
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
        }

        self.ctx.queue.submit(std::iter::once(encoder2.finish()));

        if let Some(out_buf) = output {
            // We only start reading if we have submitted at least 3 frames, to fill the pipeline
            if cache.current_frame >= 2 {
                let buffer_slice = cache.output_buffers[read_buf_index].slice(..);
                buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
                
                // Poll for the read buffer, NOT the submission index of the current frame!
                // Wait for whatever was submitted 2 frames ago.
                // Since it was submitted 2 frames ago, this poll will usually not block at all.
                let _ = self.ctx.device.poll(wgpu::PollType::Wait {
                    submission_index: None, // Just poll until everything is ready (the map_async should resolve)
                    timeout: None,
                });

                let padded_data = buffer_slice.get_mapped_range();
                for row in 0..height {
                    let start = (row * cache.padded_bytes_per_row) as usize;
                    let end = start + unpadded_bytes_per_row as usize;
                    let dst_start = (row * unpadded_bytes_per_row) as usize;
                    let dst_end = dst_start + unpadded_bytes_per_row as usize;
                    out_buf[dst_start..dst_end].copy_from_slice(&padded_data[start..end]);
                }

                drop(padded_data);
                cache.output_buffers[read_buf_index].unmap();
            }
        }
        
        cache.current_frame += 1;
    }

    pub fn get_texture_view(&self) -> Option<wgpu::TextureView> {
        let guard = self.cache.lock().unwrap();
        guard.as_ref().map(|c| c.texture.create_view(&wgpu::TextureViewDescriptor::default()))
    }
}
