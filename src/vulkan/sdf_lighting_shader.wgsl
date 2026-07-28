@group(0) @binding(0) var output_tex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var normal_coverage_tex: texture_2d<f32>;
@group(0) @binding(2) var depth_tex: texture_2d<f32>;
@group(0) @binding(3) var material_id_tex: texture_2d<u32>;

struct CameraUniform {
    pos: vec3<f32>,
    _padding0: u32,
    look_at: vec3<f32>,
    _padding1: u32,
    up: vec3<f32>,
    fov: f32,
    width: f32,
    height: f32,
    proj_type: u32,
    ortho_left: f32,
    ortho_right: f32,
    ortho_bottom: f32,
    ortho_top: f32,
    has_clip: u32,
    clip_x: f32,
    clip_y: f32,
    clip_w: f32,
    clip_h: f32,
    aa_level: u32,
    num_primitives: u32,
    raster_scale: u32,
    _pad3: u32,
    proj_mat: mat4x4<f32>,
    light_pos: vec3<f32>,
    light_intensity: f32,
    light_color: vec3<f32>,
    environment_intensity: f32,
    environment_color: vec3<f32>,
    environment_rotation: f32,
}
@group(0) @binding(4) var<uniform> camera: CameraUniform;

struct MaterialData3D {
    base_color: vec4<f32>,
    emissive: vec4<f32>,
    grid_color: vec4<f32>,
    surface: vec4<f32>,
    grid: vec4<f32>,
    grid_backface: vec4<f32>,
    transmission: vec4<f32>,
    absorption: vec4<f32>,
    patch_corner_0: vec4<f32>,
    patch_corner_1: vec4<f32>,
    patch_corner_2: vec4<f32>,
    patch_color: vec4<f32>,
    patch_edge_color: vec4<f32>,
    patch_params: vec4<f32>,
}
@group(0) @binding(5) var<storage, read> materials: array<MaterialData3D>;
@group(0) @binding(6) var environment_map: texture_2d<f32>;
@group(0) @binding(7) var environment_sampler: sampler;

fn reconstruct_position(pixel: vec2<u32>, linear_depth: f32) -> vec3<f32> {
    let forward = normalize(camera.look_at);
    let right = normalize(cross(forward, camera.up));
    let up = cross(right, forward);
    let ndc_x = ((f32(pixel.x) + 0.5) / camera.width) * 2.0 - 1.0;
    let ndc_y = 1.0 - ((f32(pixel.y) + 0.5) / camera.height) * 2.0;

    if (camera.proj_type == 0u) {
        let aspect = camera.width / camera.height;
        let fov_scale = tan(camera.fov * 0.5);
        let direction = normalize(
            right * ndc_x * aspect * fov_scale
                + up * ndc_y * fov_scale
                + forward,
        );
        return camera.pos + direction * (linear_depth / max(dot(direction, forward), 1e-5));
    }

    let horizontal = ndc_x * (camera.ortho_right - camera.ortho_left) * 0.5
        + (camera.ortho_right + camera.ortho_left) * 0.5;
    let vertical = ndc_y * (camera.ortho_top - camera.ortho_bottom) * 0.5
        + (camera.ortho_top + camera.ortho_bottom) * 0.5;
    return camera.pos + right * horizontal + up * vertical + forward * linear_depth;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dimensions = textureDimensions(output_tex);
    if (global_id.x >= dimensions.x || global_id.y >= dimensions.y) {
        return;
    }

    let pixel = vec2<i32>(global_id.xy);
    let normal_coverage = textureLoad(normal_coverage_tex, pixel, 0);
    let coverage = normal_coverage.a;
    if (coverage <= 0.0) {
        textureStore(output_tex, pixel, vec4<f32>(0.0));
        return;
    }

    let linear_depth = textureLoad(depth_tex, pixel, 0).x;
    let material_index = textureLoad(material_id_tex, pixel, 0).x;
    let material = materials[material_index];
    let position = reconstruct_position(global_id.xy, linear_depth);
    let normal = normalize(normal_coverage.xyz);
    let albedo = material.base_color.rgb;
    let roughness = clamp(material.surface.x, 0.04, 1.0);
    let metallic = clamp(material.surface.y, 0.0, 1.0);
    let reflectance_f0 = 0.16 * material.surface.z * material.surface.z;
    let f0 = mix(vec3<f32>(reflectance_f0), albedo, metallic);
    let lighting = shade_surface(
        position,
        normal,
        normalize(camera.pos - position),
        albedo,
        roughness,
        metallic,
        f0,
        material.emissive.rgb * material.emissive.a,
    );
    let alpha = material.base_color.a * coverage;
    textureStore(output_tex, pixel, vec4<f32>(lighting.color * alpha, alpha));
}
