use crate::mobjects::mesh_2d::Vertex2D;
use crate::mobjects::mesh_3d::Vertex;
use crate::vulkan::context::VulkanContext;
use ash::vk;
use std::sync::Arc;

use super::mesh_2d::Instance2D;
use super::prepared_frame::{
    GRID_AXIS_LINE_COUNT, GRID_INSTANCES_PER_GRID, GRID_LINE_COUNT, GRID_LOD_COUNT,
};

pub(super) struct PipelineSet {
    device: Arc<ash::Device>,
    pub(super) compute_descriptor_set_layout: vk::DescriptorSetLayout,
    pub(super) surface_resolve_descriptor_set_layout: vk::DescriptorSetLayout,
    pub(super) surface_lighting_descriptor_set_layout: vk::DescriptorSetLayout,
    pub(super) surface_composite_descriptor_set_layout: vk::DescriptorSetLayout,
    pub(super) raster_descriptor_set_layout: vk::DescriptorSetLayout,
    pub(super) grid_descriptor_set_layout: vk::DescriptorSetLayout,
    pub(super) raster_descriptor_set_layout_2d: vk::DescriptorSetLayout,
    pub(super) composite_descriptor_set_layout: vk::DescriptorSetLayout,
    pub(super) bloom_descriptor_set_layout: vk::DescriptorSetLayout,
    pub(super) nv12_descriptor_set_layout: vk::DescriptorSetLayout,
    pub(super) video_nv12_descriptor_set_layout: vk::DescriptorSetLayout,
    pub(super) raster_texture_layout: vk::DescriptorSetLayout,

    pub(super) compute_pipeline_layout: vk::PipelineLayout,
    pub(super) surface_resolve_pipeline_layout: vk::PipelineLayout,
    pub(super) surface_lighting_pipeline_layout: vk::PipelineLayout,
    pub(super) surface_composite_pipeline_layout: vk::PipelineLayout,
    pub(super) composite_pipeline_layout: vk::PipelineLayout,
    pub(super) bloom_pipeline_layout: vk::PipelineLayout,
    pub(super) nv12_pipeline_layout: vk::PipelineLayout,
    pub(super) video_nv12_pipeline_layout: vk::PipelineLayout,
    pub(super) raster_pipeline_layout: vk::PipelineLayout,
    pub(super) grid_pipeline_layout: vk::PipelineLayout,
    pub(super) raster_pipeline_layout_2d: vk::PipelineLayout,

    pub(super) compute_pipeline: vk::Pipeline,
    pub(super) surface_resolve_pipeline: vk::Pipeline,
    pub(super) surface_lighting_pipeline: vk::Pipeline,
    pub(super) surface_copy_pipeline: vk::Pipeline,
    pub(super) surface_overlay_pipeline: vk::Pipeline,
    pub(super) downsample_pipeline: vk::Pipeline,
    pub(super) bloom_extract_pipeline: vk::Pipeline,
    pub(super) bloom_horizontal_pipeline: vk::Pipeline,
    pub(super) bloom_vertical_pipeline: vk::Pipeline,
    pub(super) nv12_pipeline: vk::Pipeline,
    pub(super) video_nv12_pipeline: vk::Pipeline,
    pub(super) video_nv12_downsample_pipeline: vk::Pipeline,
    pub(super) yuv444p_pipeline: vk::Pipeline,
    pub(super) raster_pipeline: vk::Pipeline,
    pub(super) grid_pipeline: vk::Pipeline,
    pub(super) grid_line_width_range: [f32; 2],
    pub(super) raster_pipeline_transparent_depth: vk::Pipeline,
    pub(super) raster_pipeline_transparent_back: vk::Pipeline,
    pub(super) raster_pipeline_transparent_front: vk::Pipeline,
    pub(super) raster_pipeline_2d: vk::Pipeline,
    pub(super) raster_pipeline_2d_depthless: vk::Pipeline,
    pub(super) raster_pipeline_2d_analytic: vk::Pipeline,
}

impl PipelineSet {
    pub(super) fn new(
        ctx: &VulkanContext,
        output_transform: &str,
        sample_count: vk::SampleCountFlags,
    ) -> Self {
        let surface_interface = include_str!("../surface_interface.wgsl");
        let surface_lighting = include_str!("../surface_lighting.wgsl");
        let compute_shader_source = include_str!("../shader.wgsl");
        let raster_shader_source = format!(
            "{surface_lighting}\n{}",
            include_str!("../raster_shader.wgsl")
        );
        let raster_sample_count = sample_count.as_raw();
        let surface_gbuffer_bindings = if raster_sample_count == 1 {
            r#"
@group(0) @binding(5) var sdf_normal_coverage_tex: texture_2d<f32>;
@group(0) @binding(6) var sdf_depth_tex: texture_2d<f32>;
@group(0) @binding(7) var sdf_material_id_tex: texture_2d<u32>;
@group(0) @binding(8) var raster_normal_depth_tex: texture_2d<f32>;
@group(0) @binding(9) var raster_albedo_tex: texture_2d<f32>;
@group(0) @binding(10) var raster_material_id_tex: texture_2d<u32>;
const GBUFFER_SAMPLE_COUNT = 1u;
fn load_raster_normal_depth(pixel: vec2<i32>, sample: u32) -> vec4<f32> {
    return textureLoad(raster_normal_depth_tex, pixel, 0);
}
fn load_raster_albedo(pixel: vec2<i32>, sample: u32) -> vec4<f32> {
    return textureLoad(raster_albedo_tex, pixel, 0);
}
fn load_raster_material_id(pixel: vec2<i32>, sample: u32) -> u32 {
    return textureLoad(raster_material_id_tex, pixel, 0).x;
}
"#
            .to_owned()
        } else {
            format!(
                r#"
@group(0) @binding(5) var sdf_normal_coverage_tex: texture_2d<f32>;
@group(0) @binding(6) var sdf_depth_tex: texture_2d<f32>;
@group(0) @binding(7) var sdf_material_id_tex: texture_2d<u32>;
@group(0) @binding(8) var raster_normal_depth_tex: texture_multisampled_2d<f32>;
@group(0) @binding(9) var raster_albedo_tex: texture_multisampled_2d<f32>;
@group(0) @binding(10) var raster_material_id_tex: texture_multisampled_2d<u32>;
const GBUFFER_SAMPLE_COUNT = {raster_sample_count}u;
fn load_raster_normal_depth(pixel: vec2<i32>, sample: u32) -> vec4<f32> {{
    return textureLoad(raster_normal_depth_tex, pixel, i32(sample));
}}
fn load_raster_albedo(pixel: vec2<i32>, sample: u32) -> vec4<f32> {{
    return textureLoad(raster_albedo_tex, pixel, i32(sample));
}}
fn load_raster_material_id(pixel: vec2<i32>, sample: u32) -> u32 {{
    return textureLoad(raster_material_id_tex, pixel, i32(sample)).x;
}}
"#
            )
        };
        let surface_resolve_shader_source = format!(
            "{surface_gbuffer_bindings}\n{surface_interface}\n{}",
            include_str!("../surface_resolve_shader.wgsl")
        );
        let surface_lighting_shader_source = format!(
            "{surface_interface}\n{}\n{}",
            include_str!("../surface_lighting_shader.wgsl"),
            surface_lighting,
        );
        let compute_shader = compile_wgsl_full(ctx, compute_shader_source);
        let raster_shader = compile_wgsl_full(ctx, &raster_shader_source);
        let grid_shader_source = format!(
            "const LINE_COUNT: u32 = {GRID_LINE_COUNT}u;\n\
             const LOD_COUNT: u32 = {GRID_LOD_COUNT}u;\n\
             const AXIS_LINE_COUNT: u32 = {GRID_AXIS_LINE_COUNT}u;\n\
             const INSTANCES_PER_GRID: u32 = {GRID_INSTANCES_PER_GRID}u;\n{}",
            include_str!("../grid_shader.wgsl"),
        );
        let grid_shader = compile_wgsl_full(ctx, &grid_shader_source);
        let surface_resolve_shader = compile_wgsl_full(ctx, &surface_resolve_shader_source);
        let surface_lighting_shader = compile_wgsl_full(ctx, &surface_lighting_shader_source);
        let surface_composite_shader =
            compile_wgsl_full(ctx, include_str!("../surface_composite_shader.wgsl"));
        let raster_shader_2d = compile_wgsl_full(ctx, include_str!("../raster_shader_2d.wgsl"));
        let raster_shader_2d_analytic =
            compile_wgsl_full(ctx, include_str!("../raster_shader_2d_aa.wgsl"));
        let compile_output_shader = |source| {
            let source = format!("{output_transform}\n{source}");
            compile_wgsl_full(ctx, &source)
        };
        let nv12_shader = compile_output_shader(include_str!("../rgba_to_nv12.wgsl"));
        let video_nv12_shader = compile_output_shader(include_str!("../rgba_to_nv12_image.wgsl"));
        let video_nv12_downsample_shader =
            compile_output_shader(include_str!("../downsample_to_nv12_image.wgsl"));
        let yuv444p_shader = compile_output_shader(include_str!("../rgba_to_yuv444p.wgsl"));
        let downsample_shader = compile_output_shader(include_str!("../downsample_shader.wgsl"));
        let bloom_shader = compile_wgsl_full(ctx, include_str!("../bloom_shader.wgsl"));

        let composite_bindings = [
            vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: 1,
                descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: 2,
                descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: 3,
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            },
        ];
        let composite_layout_info = vk::DescriptorSetLayoutCreateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            binding_count: composite_bindings.len() as u32,
            p_bindings: composite_bindings.as_ptr(),
            ..Default::default()
        };
        let composite_descriptor_set_layout = unsafe {
            ctx.device
                .create_descriptor_set_layout(&composite_layout_info, None)
                .unwrap()
        };
        let bloom_bindings = [
            vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: 1,
                descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            },
        ];
        let bloom_layout_info = vk::DescriptorSetLayoutCreateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            binding_count: bloom_bindings.len() as u32,
            p_bindings: bloom_bindings.as_ptr(),
            ..Default::default()
        };
        let bloom_descriptor_set_layout = unsafe {
            ctx.device
                .create_descriptor_set_layout(&bloom_layout_info, None)
                .unwrap()
        };

        let nv12_bindings = [
            vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: 2,
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            },
        ];
        let nv12_layout_info = vk::DescriptorSetLayoutCreateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            binding_count: nv12_bindings.len() as u32,
            p_bindings: nv12_bindings.as_ptr(),
            ..Default::default()
        };
        let nv12_descriptor_set_layout = unsafe {
            ctx.device
                .create_descriptor_set_layout(&nv12_layout_info, None)
                .unwrap()
        };

        let video_nv12_bindings = [
            vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: 1,
                descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: 2,
                descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            },
        ];
        let video_nv12_layout_info = vk::DescriptorSetLayoutCreateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            binding_count: video_nv12_bindings.len() as u32,
            p_bindings: video_nv12_bindings.as_ptr(),
            ..Default::default()
        };
        let video_nv12_descriptor_set_layout = unsafe {
            ctx.device
                .create_descriptor_set_layout(&video_nv12_layout_info, None)
                .unwrap()
        };

        let nv12_pipeline_layout_info = vk::PipelineLayoutCreateInfo {
            s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
            set_layout_count: 1,
            p_set_layouts: &nv12_descriptor_set_layout,
            ..Default::default()
        };
        let nv12_pipeline_layout = unsafe {
            ctx.device
                .create_pipeline_layout(&nv12_pipeline_layout_info, None)
                .unwrap()
        };

        let main_name = std::ffi::CString::new("main").unwrap();
        let nv12_stage = vk::PipelineShaderStageCreateInfo {
            s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage: vk::ShaderStageFlags::COMPUTE,
            module: nv12_shader,
            p_name: main_name.as_ptr(),
            ..Default::default()
        };
        let nv12_pipeline_info = vk::ComputePipelineCreateInfo {
            s_type: vk::StructureType::COMPUTE_PIPELINE_CREATE_INFO,
            stage: nv12_stage,
            layout: nv12_pipeline_layout,
            ..Default::default()
        };
        let nv12_pipeline = unsafe {
            ctx.device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&nv12_pipeline_info),
                    None,
                )
                .unwrap()[0]
        };

        let video_nv12_pipeline_layout_info = vk::PipelineLayoutCreateInfo {
            s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
            set_layout_count: 1,
            p_set_layouts: &video_nv12_descriptor_set_layout,
            ..Default::default()
        };
        let video_nv12_pipeline_layout = unsafe {
            ctx.device
                .create_pipeline_layout(&video_nv12_pipeline_layout_info, None)
                .unwrap()
        };
        let video_nv12_stage = vk::PipelineShaderStageCreateInfo {
            s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage: vk::ShaderStageFlags::COMPUTE,
            module: video_nv12_shader,
            p_name: main_name.as_ptr(),
            ..Default::default()
        };
        let video_nv12_pipeline_info = vk::ComputePipelineCreateInfo {
            s_type: vk::StructureType::COMPUTE_PIPELINE_CREATE_INFO,
            stage: video_nv12_stage,
            layout: video_nv12_pipeline_layout,
            ..Default::default()
        };
        let video_nv12_pipeline = unsafe {
            ctx.device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&video_nv12_pipeline_info),
                    None,
                )
                .unwrap()[0]
        };
        let video_nv12_downsample_stage = vk::PipelineShaderStageCreateInfo {
            module: video_nv12_downsample_shader,
            ..video_nv12_stage
        };
        let video_nv12_downsample_pipeline_info = vk::ComputePipelineCreateInfo {
            stage: video_nv12_downsample_stage,
            ..video_nv12_pipeline_info
        };
        let video_nv12_downsample_pipeline = unsafe {
            ctx.device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&video_nv12_downsample_pipeline_info),
                    None,
                )
                .unwrap()[0]
        };

        let yuv444p_stage = vk::PipelineShaderStageCreateInfo {
            s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage: vk::ShaderStageFlags::COMPUTE,
            module: yuv444p_shader,
            p_name: main_name.as_ptr(),
            ..Default::default()
        };
        let yuv444p_pipeline_info = vk::ComputePipelineCreateInfo {
            s_type: vk::StructureType::COMPUTE_PIPELINE_CREATE_INFO,
            stage: yuv444p_stage,
            layout: nv12_pipeline_layout,
            ..Default::default()
        };
        let yuv444p_pipeline = unsafe {
            ctx.device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&yuv444p_pipeline_info),
                    None,
                )
                .unwrap()[0]
        };

        let composite_pipeline_layout_info = vk::PipelineLayoutCreateInfo {
            s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
            set_layout_count: 1,
            p_set_layouts: &composite_descriptor_set_layout,
            ..Default::default()
        };
        let composite_pipeline_layout = unsafe {
            ctx.device
                .create_pipeline_layout(&composite_pipeline_layout_info, None)
                .unwrap()
        };

        let downsample_stage = vk::PipelineShaderStageCreateInfo {
            s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage: vk::ShaderStageFlags::COMPUTE,
            module: downsample_shader,
            p_name: main_name.as_ptr(),
            ..Default::default()
        };
        let downsample_pipeline_info = vk::ComputePipelineCreateInfo {
            s_type: vk::StructureType::COMPUTE_PIPELINE_CREATE_INFO,
            stage: downsample_stage,
            layout: composite_pipeline_layout,
            ..Default::default()
        };
        let downsample_pipeline = unsafe {
            ctx.device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&downsample_pipeline_info),
                    None,
                )
                .unwrap()[0]
        };
        let bloom_pipeline_layout_info = vk::PipelineLayoutCreateInfo {
            s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
            set_layout_count: 1,
            p_set_layouts: &bloom_descriptor_set_layout,
            ..Default::default()
        };
        let bloom_pipeline_layout = unsafe {
            ctx.device
                .create_pipeline_layout(&bloom_pipeline_layout_info, None)
                .unwrap()
        };
        let create_bloom_pipeline = |entry_point: &str| {
            let entry_point = std::ffi::CString::new(entry_point).unwrap();
            let stage = vk::PipelineShaderStageCreateInfo {
                s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage: vk::ShaderStageFlags::COMPUTE,
                module: bloom_shader,
                p_name: entry_point.as_ptr(),
                ..Default::default()
            };
            let info = vk::ComputePipelineCreateInfo {
                s_type: vk::StructureType::COMPUTE_PIPELINE_CREATE_INFO,
                stage,
                layout: bloom_pipeline_layout,
                ..Default::default()
            };
            unsafe {
                ctx.device
                    .create_compute_pipelines(
                        vk::PipelineCache::null(),
                        std::slice::from_ref(&info),
                        None,
                    )
                    .unwrap()[0]
            }
        };
        let bloom_extract_pipeline = create_bloom_pipeline("extract");
        let bloom_horizontal_pipeline = create_bloom_pipeline("blur_horizontal");
        let bloom_vertical_pipeline = create_bloom_pipeline("blur_vertical");

        let compute_bindings = [
            vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: 1,
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE
                    | vk::ShaderStageFlags::VERTEX
                    | vk::ShaderStageFlags::FRAGMENT,
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: 2,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER_DYNAMIC,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: 3,
                descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: 4,
                descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            },
        ];

        let compute_layout_info = vk::DescriptorSetLayoutCreateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            binding_count: compute_bindings.len() as u32,
            p_bindings: compute_bindings.as_ptr(),
            ..Default::default()
        };
        let compute_descriptor_set_layout = unsafe {
            ctx.device
                .create_descriptor_set_layout(&compute_layout_info, None)
                .unwrap()
        };
        let compute_binding = |binding, descriptor_type| vk::DescriptorSetLayoutBinding {
            binding,
            descriptor_type,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            ..Default::default()
        };
        let surface_resolve_bindings = [
            compute_binding(0, vk::DescriptorType::STORAGE_IMAGE),
            compute_binding(1, vk::DescriptorType::STORAGE_IMAGE),
            compute_binding(2, vk::DescriptorType::STORAGE_IMAGE),
            compute_binding(3, vk::DescriptorType::STORAGE_IMAGE),
            compute_binding(4, vk::DescriptorType::STORAGE_IMAGE),
            compute_binding(5, vk::DescriptorType::SAMPLED_IMAGE),
            compute_binding(6, vk::DescriptorType::SAMPLED_IMAGE),
            compute_binding(7, vk::DescriptorType::SAMPLED_IMAGE),
            compute_binding(8, vk::DescriptorType::SAMPLED_IMAGE),
            compute_binding(9, vk::DescriptorType::SAMPLED_IMAGE),
            compute_binding(10, vk::DescriptorType::SAMPLED_IMAGE),
            compute_binding(11, vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC),
            compute_binding(12, vk::DescriptorType::STORAGE_BUFFER_DYNAMIC),
        ];
        let surface_resolve_descriptor_set_layout = unsafe {
            ctx.device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default()
                        .bindings(&surface_resolve_bindings),
                    None,
                )
                .unwrap()
        };
        let surface_lighting_bindings = [
            compute_binding(0, vk::DescriptorType::STORAGE_IMAGE),
            compute_binding(1, vk::DescriptorType::SAMPLED_IMAGE),
            compute_binding(2, vk::DescriptorType::SAMPLED_IMAGE),
            compute_binding(3, vk::DescriptorType::SAMPLED_IMAGE),
            compute_binding(4, vk::DescriptorType::SAMPLED_IMAGE),
            compute_binding(5, vk::DescriptorType::SAMPLED_IMAGE),
            compute_binding(6, vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC),
            compute_binding(7, vk::DescriptorType::STORAGE_BUFFER_DYNAMIC),
            compute_binding(8, vk::DescriptorType::SAMPLED_IMAGE),
            compute_binding(9, vk::DescriptorType::SAMPLER),
        ];
        let surface_lighting_descriptor_set_layout = unsafe {
            ctx.device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default()
                        .bindings(&surface_lighting_bindings),
                    None,
                )
                .unwrap()
        };
        let surface_composite_bindings = [
            vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: 1,
                descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: 2,
                descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            },
        ];
        let surface_composite_descriptor_set_layout = unsafe {
            ctx.device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default()
                        .bindings(&surface_composite_bindings),
                    None,
                )
                .unwrap()
        };

        let raster_bindings = [
            vk::DescriptorSetLayoutBinding {
                binding: 1,
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: 2,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER_DYNAMIC,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::FRAGMENT,
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: 3,
                descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::FRAGMENT,
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: 4,
                descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::FRAGMENT,
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: 5,
                descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::FRAGMENT,
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: 6,
                descriptor_type: vk::DescriptorType::SAMPLER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::FRAGMENT,
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: 7,
                descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::FRAGMENT,
                ..Default::default()
            },
        ];
        let raster_layout_info = vk::DescriptorSetLayoutCreateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            binding_count: raster_bindings.len() as u32,
            p_bindings: raster_bindings.as_ptr(),
            ..Default::default()
        };
        let raster_descriptor_set_layout = unsafe {
            ctx.device
                .create_descriptor_set_layout(&raster_layout_info, None)
                .unwrap()
        };

        let grid_bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER_DYNAMIC)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        ];
        let grid_descriptor_set_layout = unsafe {
            ctx.device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&grid_bindings),
                    None,
                )
                .unwrap()
        };

        let compute_pipeline_layout_info = vk::PipelineLayoutCreateInfo {
            s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
            set_layout_count: 1,
            p_set_layouts: &compute_descriptor_set_layout,
            ..Default::default()
        };
        let compute_pipeline_layout = unsafe {
            ctx.device
                .create_pipeline_layout(&compute_pipeline_layout_info, None)
                .unwrap()
        };

        let main_name = std::ffi::CString::new("main").unwrap();
        let compute_stage = vk::PipelineShaderStageCreateInfo {
            s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage: vk::ShaderStageFlags::COMPUTE,
            module: compute_shader,
            p_name: main_name.as_ptr(),
            ..Default::default()
        };

        let compute_pipeline_info = vk::ComputePipelineCreateInfo {
            s_type: vk::StructureType::COMPUTE_PIPELINE_CREATE_INFO,
            stage: compute_stage,
            layout: compute_pipeline_layout,
            ..Default::default()
        };
        let compute_pipeline = unsafe {
            ctx.device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&compute_pipeline_info),
                    None,
                )
                .unwrap()[0]
        };
        let surface_resolve_pipeline_layout = unsafe {
            ctx.device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(std::slice::from_ref(&surface_resolve_descriptor_set_layout)),
                    None,
                )
                .unwrap()
        };
        let create_compute_pipeline = |shader: vk::ShaderModule, layout: vk::PipelineLayout| unsafe {
            let entry_point = std::ffi::CString::new("main").unwrap();
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(shader)
                .name(&entry_point);
            ctx.device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(
                        &vk::ComputePipelineCreateInfo::default()
                            .stage(stage)
                            .layout(layout),
                    ),
                    None,
                )
                .unwrap()[0]
        };
        let surface_resolve_pipeline =
            create_compute_pipeline(surface_resolve_shader, surface_resolve_pipeline_layout);
        let surface_lighting_pipeline_layout = unsafe {
            ctx.device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().set_layouts(std::slice::from_ref(
                        &surface_lighting_descriptor_set_layout,
                    )),
                    None,
                )
                .unwrap()
        };
        let surface_lighting_pipeline =
            create_compute_pipeline(surface_lighting_shader, surface_lighting_pipeline_layout);
        let surface_composite_pipeline_layout = unsafe {
            ctx.device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().set_layouts(std::slice::from_ref(
                        &surface_composite_descriptor_set_layout,
                    )),
                    None,
                )
                .unwrap()
        };
        let create_surface_composite_pipeline = |entry_point: &str| {
            let entry_point = std::ffi::CString::new(entry_point).unwrap();
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(surface_composite_shader)
                .name(&entry_point);
            unsafe {
                ctx.device
                    .create_compute_pipelines(
                        vk::PipelineCache::null(),
                        std::slice::from_ref(
                            &vk::ComputePipelineCreateInfo::default()
                                .stage(stage)
                                .layout(surface_composite_pipeline_layout),
                        ),
                        None,
                    )
                    .unwrap()[0]
            }
        };
        let surface_copy_pipeline = create_surface_composite_pipeline("copy_surface");
        let surface_overlay_pipeline = create_surface_composite_pipeline("composite_overlay");

        let raster_pipeline_layout_info = vk::PipelineLayoutCreateInfo {
            s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
            set_layout_count: 1,
            p_set_layouts: &raster_descriptor_set_layout,
            ..Default::default()
        };
        let raster_pipeline_layout = unsafe {
            ctx.device
                .create_pipeline_layout(&raster_pipeline_layout_info, None)
                .unwrap()
        };
        let grid_pipeline_layout = unsafe {
            ctx.device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(std::slice::from_ref(&grid_descriptor_set_layout)),
                    None,
                )
                .unwrap()
        };

        let raster_descriptor_set_layout_bindings_2d = [vk::DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            p_immutable_samplers: std::ptr::null(),
            ..Default::default()
        }];

        let texture_layout_bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(16)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        ];
        let binding_flags = [
            vk::DescriptorBindingFlags::PARTIALLY_BOUND,
            vk::DescriptorBindingFlags::empty(),
        ];
        let mut layout_binding_flags =
            vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&binding_flags);

        let texture_layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&texture_layout_bindings)
            .push_next(&mut layout_binding_flags);

        let raster_texture_layout = unsafe {
            ctx.device
                .create_descriptor_set_layout(&texture_layout_info, None)
                .unwrap()
        };

        let raster_descriptor_set_layout_info_2d = vk::DescriptorSetLayoutCreateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            binding_count: raster_descriptor_set_layout_bindings_2d.len() as u32,
            p_bindings: raster_descriptor_set_layout_bindings_2d.as_ptr(),
            ..Default::default()
        };
        let raster_descriptor_set_layout_2d = unsafe {
            ctx.device
                .create_descriptor_set_layout(&raster_descriptor_set_layout_info_2d, None)
                .unwrap()
        };

        let set_layouts_2d = [raster_descriptor_set_layout_2d, raster_texture_layout];
        let raster_pipeline_layout_info_2d =
            vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts_2d);
        let raster_pipeline_layout_2d = unsafe {
            ctx.device
                .create_pipeline_layout(&raster_pipeline_layout_info_2d, None)
                .unwrap()
        };

        let vertex_input_binding = vk::VertexInputBindingDescription {
            binding: 0,
            stride: std::mem::size_of::<Vertex>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        };

        let vertex_input_attributes = [
            vk::VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 0,
            },
            vk::VertexInputAttributeDescription {
                location: 1,
                binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 12,
            },
            vk::VertexInputAttributeDescription {
                location: 2,
                binding: 0,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: 24,
            },
            vk::VertexInputAttributeDescription {
                location: 3,
                binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 40,
            },
        ];

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            vertex_binding_description_count: 1,
            p_vertex_binding_descriptions: &vertex_input_binding,
            vertex_attribute_description_count: vertex_input_attributes.len() as u32,
            p_vertex_attribute_descriptions: vertex_input_attributes.as_ptr(),
            ..Default::default()
        };

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            primitive_restart_enable: vk::FALSE,
            ..Default::default()
        };

        let viewport_state = vk::PipelineViewportStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            viewport_count: 1,
            scissor_count: 1,
            ..Default::default()
        };

        let rasterizer = vk::PipelineRasterizationStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            depth_clamp_enable: vk::FALSE,
            rasterizer_discard_enable: vk::FALSE,
            polygon_mode: vk::PolygonMode::FILL,
            line_width: 1.0,
            cull_mode: vk::CullModeFlags::NONE,
            front_face: vk::FrontFace::CLOCKWISE, // naga flips Y, which reverses winding order
            depth_bias_enable: vk::FALSE,
            ..Default::default()
        };

        let multisampling = vk::PipelineMultisampleStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            sample_shading_enable: vk::FALSE,
            rasterization_samples: sample_count,
            ..Default::default()
        };

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            depth_test_enable: vk::TRUE,
            depth_write_enable: vk::TRUE,
            depth_compare_op: vk::CompareOp::LESS,
            depth_bounds_test_enable: vk::FALSE,
            stencil_test_enable: vk::FALSE,
            ..Default::default()
        };

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState {
            color_write_mask: vk::ColorComponentFlags::RGBA,
            blend_enable: vk::FALSE,
            src_color_blend_factor: vk::BlendFactor::SRC_ALPHA,
            dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
            color_blend_op: vk::BlendOp::ADD,
            src_alpha_blend_factor: vk::BlendFactor::ONE,
            dst_alpha_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
            alpha_blend_op: vk::BlendOp::ADD,
        };

        let color_blending = vk::PipelineColorBlendStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            logic_op_enable: vk::FALSE,
            attachment_count: 1,
            p_attachments: &color_blend_attachment,
            ..Default::default()
        };

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state = vk::PipelineDynamicStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            dynamic_state_count: dynamic_states.len() as u32,
            p_dynamic_states: dynamic_states.as_ptr(),
            ..Default::default()
        };

        let vs_name = std::ffi::CString::new("vs_main").unwrap();
        let fs_name = std::ffi::CString::new("fs_main").unwrap();

        let shader_stages = [
            vk::PipelineShaderStageCreateInfo {
                s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage: vk::ShaderStageFlags::VERTEX,
                module: raster_shader,
                p_name: vs_name.as_ptr(),
                ..Default::default()
            },
            vk::PipelineShaderStageCreateInfo {
                s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage: vk::ShaderStageFlags::FRAGMENT,
                module: raster_shader,
                p_name: fs_name.as_ptr(),
                ..Default::default()
            },
        ];
        let raster_color_formats = [vk::Format::R16G16B16A16_SFLOAT];
        let pipeline_rendering_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&raster_color_formats)
            .depth_attachment_format(vk::Format::D32_SFLOAT);

        let raster_pipeline_info = vk::GraphicsPipelineCreateInfo {
            s_type: vk::StructureType::GRAPHICS_PIPELINE_CREATE_INFO,
            p_next: (&pipeline_rendering_info as *const vk::PipelineRenderingCreateInfo).cast(),
            stage_count: shader_stages.len() as u32,
            p_stages: shader_stages.as_ptr(),
            p_vertex_input_state: &vertex_input_info,
            p_input_assembly_state: &input_assembly,
            p_viewport_state: &viewport_state,
            p_rasterization_state: &rasterizer,
            p_multisample_state: &multisampling,
            p_depth_stencil_state: &depth_stencil,
            p_color_blend_state: &color_blending,
            p_dynamic_state: &dynamic_state,
            layout: raster_pipeline_layout,
            render_pass: vk::RenderPass::null(),
            ..Default::default()
        };

        let gbuffer_fs_name = std::ffi::CString::new("fs_gbuffer").unwrap();
        let gbuffer_shader_stages = [
            shader_stages[0],
            vk::PipelineShaderStageCreateInfo {
                p_name: gbuffer_fs_name.as_ptr(),
                ..shader_stages[1]
            },
        ];
        let gbuffer_color_formats = [
            vk::Format::R16G16B16A16_SFLOAT,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::Format::R16_UINT,
        ];
        let gbuffer_rendering_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&gbuffer_color_formats)
            .depth_attachment_format(vk::Format::D32_SFLOAT);
        let gbuffer_blend_attachments = [color_blend_attachment; 3];
        let gbuffer_color_blending = vk::PipelineColorBlendStateCreateInfo {
            attachment_count: gbuffer_blend_attachments.len() as u32,
            p_attachments: gbuffer_blend_attachments.as_ptr(),
            ..color_blending
        };
        let gbuffer_pipeline_info = vk::GraphicsPipelineCreateInfo {
            p_next: (&gbuffer_rendering_info as *const vk::PipelineRenderingCreateInfo).cast(),
            stage_count: gbuffer_shader_stages.len() as u32,
            p_stages: gbuffer_shader_stages.as_ptr(),
            p_color_blend_state: &gbuffer_color_blending,
            ..raster_pipeline_info
        };
        let raster_pipeline = unsafe {
            ctx.device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&gbuffer_pipeline_info),
                    None,
                )
                .unwrap()[0]
        };
        let grid_shader_stages = [
            vk::PipelineShaderStageCreateInfo {
                module: grid_shader,
                ..shader_stages[0]
            },
            vk::PipelineShaderStageCreateInfo {
                module: grid_shader,
                ..shader_stages[1]
            },
        ];
        let grid_vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let grid_input_assembly = vk::PipelineInputAssemblyStateCreateInfo {
            topology: vk::PrimitiveTopology::LINE_LIST,
            ..input_assembly
        };
        let grid_line_rasterization = vk::PipelineRasterizationLineStateCreateInfoEXT::default()
            .line_rasterization_mode(vk::LineRasterizationModeEXT::RECTANGULAR);
        let grid_rasterizer = vk::PipelineRasterizationStateCreateInfo {
            p_next: (&grid_line_rasterization
                as *const vk::PipelineRasterizationLineStateCreateInfoEXT)
                .cast(),
            ..rasterizer
        };
        let grid_dynamic_states = [
            vk::DynamicState::VIEWPORT,
            vk::DynamicState::SCISSOR,
            vk::DynamicState::LINE_WIDTH,
        ];
        let grid_dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&grid_dynamic_states);
        let grid_depth_stencil = vk::PipelineDepthStencilStateCreateInfo {
            depth_write_enable: vk::FALSE,
            depth_compare_op: vk::CompareOp::LESS_OR_EQUAL,
            ..depth_stencil
        };
        // Grid lines are emissive strokes of a shared color: compositing two
        // crossing lines with alpha-over would make every crossing brighter
        // than the lines themselves. The fragment shader emits premultiplied
        // color, and a MAX blend keeps crossings at single-line brightness.
        let grid_blend_attachment = vk::PipelineColorBlendAttachmentState {
            blend_enable: vk::TRUE,
            src_color_blend_factor: vk::BlendFactor::ONE,
            dst_color_blend_factor: vk::BlendFactor::ONE,
            color_blend_op: vk::BlendOp::MAX,
            src_alpha_blend_factor: vk::BlendFactor::ONE,
            dst_alpha_blend_factor: vk::BlendFactor::ONE,
            alpha_blend_op: vk::BlendOp::MAX,
            ..color_blend_attachment
        };
        let grid_color_blending = vk::PipelineColorBlendStateCreateInfo {
            p_attachments: &grid_blend_attachment,
            ..color_blending
        };
        let grid_pipeline_info = vk::GraphicsPipelineCreateInfo {
            stage_count: grid_shader_stages.len() as u32,
            p_stages: grid_shader_stages.as_ptr(),
            p_vertex_input_state: &grid_vertex_input,
            p_input_assembly_state: &grid_input_assembly,
            p_rasterization_state: &grid_rasterizer,
            p_depth_stencil_state: &grid_depth_stencil,
            p_color_blend_state: &grid_color_blending,
            p_dynamic_state: &grid_dynamic_state,
            layout: grid_pipeline_layout,
            ..raster_pipeline_info
        };
        let grid_pipeline = unsafe {
            ctx.device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&grid_pipeline_info),
                    None,
                )
                .unwrap()[0]
        };
        let transparent_depth_stencil = vk::PipelineDepthStencilStateCreateInfo {
            depth_write_enable: vk::FALSE,
            ..depth_stencil
        };
        let transparent_blend_attachment = vk::PipelineColorBlendAttachmentState {
            blend_enable: vk::TRUE,
            ..color_blend_attachment
        };
        let transparent_color_blending = vk::PipelineColorBlendStateCreateInfo {
            p_attachments: &transparent_blend_attachment,
            ..color_blending
        };
        let transparent_back_rasterizer = vk::PipelineRasterizationStateCreateInfo {
            cull_mode: vk::CullModeFlags::FRONT,
            ..rasterizer
        };
        let transparent_front_rasterizer = vk::PipelineRasterizationStateCreateInfo {
            cull_mode: vk::CullModeFlags::BACK,
            ..rasterizer
        };
        let thickness_fs_name = std::ffi::CString::new("fs_back_depth").unwrap();
        let thickness_shader_stages = [
            shader_stages[0],
            vk::PipelineShaderStageCreateInfo {
                p_name: thickness_fs_name.as_ptr(),
                ..shader_stages[1]
            },
        ];
        let thickness_multisampling = vk::PipelineMultisampleStateCreateInfo {
            rasterization_samples: vk::SampleCountFlags::TYPE_1,
            ..multisampling
        };
        let thickness_depth_stencil = vk::PipelineDepthStencilStateCreateInfo {
            depth_test_enable: vk::FALSE,
            depth_write_enable: vk::FALSE,
            depth_compare_op: vk::CompareOp::ALWAYS,
            ..depth_stencil
        };
        let thickness_blend_attachment = vk::PipelineColorBlendAttachmentState {
            color_write_mask: vk::ColorComponentFlags::R,
            blend_enable: vk::TRUE,
            src_color_blend_factor: vk::BlendFactor::ONE,
            dst_color_blend_factor: vk::BlendFactor::ONE,
            color_blend_op: vk::BlendOp::MAX,
            src_alpha_blend_factor: vk::BlendFactor::ONE,
            dst_alpha_blend_factor: vk::BlendFactor::ONE,
            alpha_blend_op: vk::BlendOp::MAX,
        };
        let thickness_color_blending = vk::PipelineColorBlendStateCreateInfo {
            p_attachments: &thickness_blend_attachment,
            ..color_blending
        };
        let thickness_color_formats = [vk::Format::R32_SFLOAT];
        let thickness_rendering_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&thickness_color_formats);
        let thickness_pipeline_info = vk::GraphicsPipelineCreateInfo {
            p_next: (&thickness_rendering_info as *const vk::PipelineRenderingCreateInfo).cast(),
            stage_count: thickness_shader_stages.len() as u32,
            p_stages: thickness_shader_stages.as_ptr(),
            p_rasterization_state: &transparent_back_rasterizer,
            p_multisample_state: &thickness_multisampling,
            p_depth_stencil_state: &thickness_depth_stencil,
            p_color_blend_state: &thickness_color_blending,
            ..raster_pipeline_info
        };
        let raster_pipeline_transparent_depth = unsafe {
            ctx.device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&thickness_pipeline_info),
                    None,
                )
                .unwrap()[0]
        };
        let transparent_back_info = vk::GraphicsPipelineCreateInfo {
            p_rasterization_state: &transparent_back_rasterizer,
            p_depth_stencil_state: &transparent_depth_stencil,
            p_color_blend_state: &transparent_color_blending,
            ..raster_pipeline_info
        };
        let transparent_front_info = vk::GraphicsPipelineCreateInfo {
            p_rasterization_state: &transparent_front_rasterizer,
            p_depth_stencil_state: &transparent_depth_stencil,
            p_color_blend_state: &transparent_color_blending,
            ..raster_pipeline_info
        };
        let transparent_pipelines = unsafe {
            ctx.device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &[transparent_back_info, transparent_front_info],
                    None,
                )
                .unwrap()
        };
        let raster_pipeline_transparent_back = transparent_pipelines[0];
        let raster_pipeline_transparent_front = transparent_pipelines[1];

        let vertex_binding_descriptions_2d = [
            vk::VertexInputBindingDescription {
                binding: 0,
                stride: std::mem::size_of::<Vertex2D>() as u32,
                input_rate: vk::VertexInputRate::VERTEX,
            },
            vk::VertexInputBindingDescription {
                binding: 1,
                stride: std::mem::size_of::<Instance2D>() as u32,
                input_rate: vk::VertexInputRate::INSTANCE,
            },
        ];
        let vertex_attribute_descriptions_2d = [
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: memoffset::offset_of!(Vertex2D, position) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 6,
                format: vk::Format::R32G32_SFLOAT,
                offset: memoffset::offset_of!(Vertex2D, local) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding: 1,
                location: 1,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: memoffset::offset_of!(Instance2D, model_0) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding: 1,
                location: 2,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: memoffset::offset_of!(Instance2D, model_1) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding: 1,
                location: 3,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: memoffset::offset_of!(Instance2D, model_2) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding: 1,
                location: 4,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: memoffset::offset_of!(Instance2D, model_3) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding: 1,
                location: 5,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: memoffset::offset_of!(Instance2D, color) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding: 1,
                location: 7,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: memoffset::offset_of!(Instance2D, aa_params) as u32,
            },
        ];
        let vertex_input_info_2d = vk::PipelineVertexInputStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            vertex_binding_description_count: vertex_binding_descriptions_2d.len() as u32,
            p_vertex_binding_descriptions: vertex_binding_descriptions_2d.as_ptr(),
            vertex_attribute_description_count: vertex_attribute_descriptions_2d.len() as u32,
            p_vertex_attribute_descriptions: vertex_attribute_descriptions_2d.as_ptr(),
            ..Default::default()
        };

        let shader_stages_2d = [
            vk::PipelineShaderStageCreateInfo {
                s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage: vk::ShaderStageFlags::VERTEX,
                module: raster_shader_2d,
                p_name: vs_name.as_ptr(),
                ..Default::default()
            },
            vk::PipelineShaderStageCreateInfo {
                s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage: vk::ShaderStageFlags::FRAGMENT,
                module: raster_shader_2d,
                p_name: fs_name.as_ptr(),
                ..Default::default()
            },
        ];

        let depth_stencil_2d = vk::PipelineDepthStencilStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            depth_test_enable: vk::FALSE, // 2D overlays without depth testing (painter's algorithm)
            depth_write_enable: vk::FALSE,
            depth_compare_op: vk::CompareOp::ALWAYS,
            depth_bounds_test_enable: vk::FALSE,
            stencil_test_enable: vk::FALSE,
            ..Default::default()
        };

        let raster_pipeline_info_2d = vk::GraphicsPipelineCreateInfo {
            s_type: vk::StructureType::GRAPHICS_PIPELINE_CREATE_INFO,
            p_next: (&pipeline_rendering_info as *const vk::PipelineRenderingCreateInfo).cast(),
            stage_count: shader_stages_2d.len() as u32,
            p_stages: shader_stages_2d.as_ptr(),
            p_vertex_input_state: &vertex_input_info_2d,
            p_input_assembly_state: &input_assembly,
            p_viewport_state: &viewport_state,
            p_rasterization_state: &rasterizer,
            p_multisample_state: &multisampling,
            p_depth_stencil_state: &depth_stencil_2d,
            p_color_blend_state: &transparent_color_blending,
            p_dynamic_state: &dynamic_state,
            layout: raster_pipeline_layout_2d,
            render_pass: vk::RenderPass::null(),
            ..Default::default()
        };

        let raster_pipeline_2d = unsafe {
            ctx.device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&raster_pipeline_info_2d),
                    None,
                )
                .unwrap()[0]
        };
        let depthless_pipeline_rendering_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&raster_color_formats);
        let raster_pipeline_info_2d_depthless = vk::GraphicsPipelineCreateInfo {
            p_next: (&depthless_pipeline_rendering_info as *const vk::PipelineRenderingCreateInfo)
                .cast(),
            ..raster_pipeline_info_2d
        };
        let raster_pipeline_2d_depthless = unsafe {
            ctx.device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&raster_pipeline_info_2d_depthless),
                    None,
                )
                .unwrap()[0]
        };
        // Analytic-AA 2D raster: one sample per pixel at output resolution,
        // edge coverage derived from rect-local coordinates in the fragment
        // shader. Same attachment format and blend state as the depthless
        // pipeline so the tone-map pass is the single shared output path.
        let analytic_shader_stages_2d = [
            vk::PipelineShaderStageCreateInfo {
                module: raster_shader_2d_analytic,
                ..shader_stages_2d[0]
            },
            vk::PipelineShaderStageCreateInfo {
                module: raster_shader_2d_analytic,
                ..shader_stages_2d[1]
            },
        ];
        let analytic_multisampling = vk::PipelineMultisampleStateCreateInfo {
            rasterization_samples: vk::SampleCountFlags::TYPE_1,
            ..multisampling
        };
        let raster_pipeline_info_2d_analytic = vk::GraphicsPipelineCreateInfo {
            p_next: (&depthless_pipeline_rendering_info as *const vk::PipelineRenderingCreateInfo)
                .cast(),
            stage_count: analytic_shader_stages_2d.len() as u32,
            p_stages: analytic_shader_stages_2d.as_ptr(),
            p_multisample_state: &analytic_multisampling,
            ..raster_pipeline_info_2d
        };
        let raster_pipeline_2d_analytic = unsafe {
            ctx.device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&raster_pipeline_info_2d_analytic),
                    None,
                )
                .unwrap()[0]
        };

        unsafe {
            ctx.device.destroy_shader_module(compute_shader, None);
            ctx.device.destroy_shader_module(raster_shader, None);
            ctx.device.destroy_shader_module(grid_shader, None);
            ctx.device
                .destroy_shader_module(surface_resolve_shader, None);
            ctx.device
                .destroy_shader_module(surface_lighting_shader, None);
            ctx.device
                .destroy_shader_module(surface_composite_shader, None);
            ctx.device.destroy_shader_module(raster_shader_2d, None);
            ctx.device
                .destroy_shader_module(raster_shader_2d_analytic, None);
            ctx.device.destroy_shader_module(nv12_shader, None);
            ctx.device.destroy_shader_module(video_nv12_shader, None);
            ctx.device
                .destroy_shader_module(video_nv12_downsample_shader, None);
            ctx.device.destroy_shader_module(yuv444p_shader, None);
            ctx.device.destroy_shader_module(downsample_shader, None);
            ctx.device.destroy_shader_module(bloom_shader, None);
        }

        Self {
            device: Arc::clone(&ctx.device),
            compute_descriptor_set_layout,
            surface_resolve_descriptor_set_layout,
            surface_lighting_descriptor_set_layout,
            surface_composite_descriptor_set_layout,
            raster_descriptor_set_layout,
            grid_descriptor_set_layout,
            raster_descriptor_set_layout_2d,
            composite_descriptor_set_layout,
            bloom_descriptor_set_layout,
            nv12_descriptor_set_layout,
            video_nv12_descriptor_set_layout,
            raster_texture_layout,
            compute_pipeline_layout,
            surface_resolve_pipeline_layout,
            surface_lighting_pipeline_layout,
            surface_composite_pipeline_layout,
            composite_pipeline_layout,
            bloom_pipeline_layout,
            nv12_pipeline_layout,
            video_nv12_pipeline_layout,
            raster_pipeline_layout,
            grid_pipeline_layout,
            raster_pipeline_layout_2d,
            compute_pipeline,
            surface_resolve_pipeline,
            surface_lighting_pipeline,
            surface_copy_pipeline,
            surface_overlay_pipeline,
            downsample_pipeline,
            bloom_extract_pipeline,
            bloom_horizontal_pipeline,
            bloom_vertical_pipeline,
            nv12_pipeline,
            video_nv12_pipeline,
            video_nv12_downsample_pipeline,
            yuv444p_pipeline,
            raster_pipeline,
            grid_pipeline,
            grid_line_width_range: unsafe {
                ctx.instance
                    .get_physical_device_properties(ctx.physical_device)
                    .limits
                    .line_width_range
            },
            raster_pipeline_transparent_depth,
            raster_pipeline_transparent_back,
            raster_pipeline_transparent_front,
            raster_pipeline_2d,
            raster_pipeline_2d_depthless,
            raster_pipeline_2d_analytic,
        }
    }
}

impl Drop for PipelineSet {
    fn drop(&mut self) {
        unsafe {
            for pipeline in [
                self.compute_pipeline,
                self.surface_resolve_pipeline,
                self.surface_lighting_pipeline,
                self.surface_copy_pipeline,
                self.surface_overlay_pipeline,
                self.downsample_pipeline,
                self.bloom_extract_pipeline,
                self.bloom_horizontal_pipeline,
                self.bloom_vertical_pipeline,
                self.nv12_pipeline,
                self.video_nv12_pipeline,
                self.video_nv12_downsample_pipeline,
                self.yuv444p_pipeline,
                self.raster_pipeline,
                self.grid_pipeline,
                self.raster_pipeline_transparent_depth,
                self.raster_pipeline_transparent_back,
                self.raster_pipeline_transparent_front,
                self.raster_pipeline_2d,
                self.raster_pipeline_2d_depthless,
                self.raster_pipeline_2d_analytic,
            ] {
                self.device.destroy_pipeline(pipeline, None);
            }
            for layout in [
                self.compute_pipeline_layout,
                self.surface_resolve_pipeline_layout,
                self.surface_lighting_pipeline_layout,
                self.surface_composite_pipeline_layout,
                self.composite_pipeline_layout,
                self.bloom_pipeline_layout,
                self.nv12_pipeline_layout,
                self.video_nv12_pipeline_layout,
                self.raster_pipeline_layout,
                self.grid_pipeline_layout,
                self.raster_pipeline_layout_2d,
            ] {
                self.device.destroy_pipeline_layout(layout, None);
            }
            for layout in [
                self.compute_descriptor_set_layout,
                self.surface_resolve_descriptor_set_layout,
                self.surface_lighting_descriptor_set_layout,
                self.surface_composite_descriptor_set_layout,
                self.raster_descriptor_set_layout,
                self.grid_descriptor_set_layout,
                self.raster_descriptor_set_layout_2d,
                self.composite_descriptor_set_layout,
                self.bloom_descriptor_set_layout,
                self.nv12_descriptor_set_layout,
                self.video_nv12_descriptor_set_layout,
                self.raster_texture_layout,
            ] {
                self.device.destroy_descriptor_set_layout(layout, None);
            }
        }
    }
}

fn compile_wgsl_full(ctx: &VulkanContext, source: &str) -> vk::ShaderModule {
    let module = naga::front::wgsl::parse_str(source).unwrap();
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    let info = validator.validate(&module).unwrap();
    let options = naga::back::spv::Options::default();
    let spv = naga::back::spv::write_vec(&module, &info, &options, None).unwrap();
    let create_info = vk::ShaderModuleCreateInfo {
        s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO,
        code_size: spv.len() * 4,
        p_code: spv.as_ptr(),
        ..Default::default()
    };
    unsafe { ctx.device.create_shader_module(&create_info, None).unwrap() }
}
