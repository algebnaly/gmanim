// Analytic-AA 2D raster shader.
//
// For rectangle instances the geometry stores per-vertex coordinates in the
// rectangle's own frame (edges at +/- half_extents), so the fragment shader
// can convert the distance to the nearest edge into pixel coverage via
// fwidth. This replaces MSAA/SSAA for 2D-only frames: the raster target runs
// at output resolution with a single sample.
//
// Instances without the AA flag (generic path meshes) render without
// coverage modulation; such frames never select this pipeline because the
// renderer only enables the analytic plan when every instance is a
// rectangle.

struct CameraUniform2D {
    width: f32,
    height: f32,
    scale_factor: f32,
    _pad: f32,
}
@group(0) @binding(0) var<uniform> camera: CameraUniform2D;

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
    out.clip_position = vec4<f32>(x_ndc, y_ndc, 0.5, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var alpha = in.color.a;
    if (in.aa_mode > 0.5) {
        // Signed distance to each rect edge in local units, converted to
        // pixels through the screen-space gradient of the interpolated local
        // coordinates. `distance + 0.5` approximates box-filtered pixel
        // coverage: fully covered pixels keep alpha, edge pixels fade.
        let distance_x = in.half_extents.x - abs(in.local.x);
        let distance_y = in.half_extents.y - abs(in.local.y);
        let pixel_distance_x = distance_x / max(fwidth(in.local.x), 1e-5);
        let pixel_distance_y = distance_y / max(fwidth(in.local.y), 1e-5);
        let coverage = clamp(min(pixel_distance_x, pixel_distance_y) + 0.5, 0.0, 1.0);
        alpha *= coverage;
    }
    return vec4<f32>(in.color.rgb, alpha);
}
