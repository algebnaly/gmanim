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
    var out: VertexOutput;
    
    // Convert from pixel coordinates to NDC
    // Usually, 2D origin is center. x: [-width/2, width/2], y: [-height/2, height/2]
    // Orthographic projection:
    let x_ndc = model.position.x / (camera.width / 2.0);
    let y_ndc = model.position.y / (camera.height / 2.0);
    
    // In Vulkan, Y goes down, but Manim usually maps +Y to UP, so we flip Y if necessary.
    out.clip_position = vec4<f32>(x_ndc, -y_ndc, 0.5, 1.0);
    out.color = model.color;
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color.rgb, in.color.a);
}
