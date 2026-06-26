struct CameraUniform2D {
    width: f32,
    height: f32,
    scale_factor: f32,
    _pad: f32,
}
@group(0) @binding(0) var<uniform> camera: CameraUniform2D;

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    let world_w = camera.width / camera.scale_factor;
    let world_h = camera.height / camera.scale_factor;

    let x_ndc = model.position.x / (world_w / 2.0);
    let y_ndc = model.position.y / (world_h / 2.0);

    var out: VertexOutput;
    out.color = model.color;
    
    // In Vulkan, Y goes down, but Manim usually maps +Y to UP, so we flip Y
    out.clip_position = vec4<f32>(x_ndc, -y_ndc, 0.5, 1.0);
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color.rgb, in.color.a);
}
