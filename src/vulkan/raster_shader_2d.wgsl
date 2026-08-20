struct CameraUniform2D {
    width: f32,
    height: f32,
    scale_factor: f32,
    _pad: f32,
}
@group(0) @binding(0) var<uniform> camera: CameraUniform2D;
@group(1) @binding(0) var scene_textures: binding_array<texture_2d<f32>>;
@group(1) @binding(1) var scene_sampler: sampler;

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) model_0: vec4<f32>,
    @location(2) model_1: vec4<f32>,
    @location(3) model_2: vec4<f32>,
    @location(4) model_3: vec4<f32>,
    @location(5) color: vec4<f32>,
    @location(6) local: vec2<f32>,
    @location(7) aa_params: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) local: vec2<f32>,
    @location(2) aa_mode: f32,
    @location(3) half_extents: vec2<f32>,
}

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    let world_w = camera.width / camera.scale_factor;
    let world_h = camera.height / camera.scale_factor;
    let model_matrix = mat4x4<f32>(
        model.model_0,
        model.model_1,
        model.model_2,
        model.model_3,
    );
    let world_position = model_matrix * vec4<f32>(model.position, 0.0, 1.0);

    let x_ndc = world_position.x / (world_w / 2.0);
    let y_ndc = world_position.y / (world_h / 2.0);

    var out: VertexOutput;
    out.color = model.color;
    out.local = model.local;
    out.aa_mode = model.aa_params.z;
    out.half_extents = model.aa_params.xy;
    
    // wgpu automatically handles coordinate system mapping
    out.clip_position = vec4<f32>(x_ndc, y_ndc, 0.5, 1.0);
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if (in.aa_mode >= 2.0 && in.aa_mode < 18.0) {
        let tex_idx = i32(in.aa_mode) - 2;
        let u = (in.local.y / max(in.half_extents.y, 1e-5)) * 0.5 + 0.5;
        let v = (in.local.x / max(in.half_extents.x, 1e-5)) * 0.5 + 0.5;
        let uv = clamp(vec2<f32>(u, v), vec2<f32>(0.0), vec2<f32>(1.0));
        let tex_color = textureSample(scene_textures[tex_idx], scene_sampler, uv);
        return tex_color * vec4<f32>(1.0, 1.0, 1.0, in.color.a);
    }
    return vec4<f32>(in.color.rgb, in.color.a);
}
