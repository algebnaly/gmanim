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
    has_raster_surfaces: u32,
    proj_mat: mat4x4<f32>,
    light_pos: vec3<f32>,
    light_intensity: f32,
    light_color: vec3<f32>,
    environment_intensity: f32,
    environment_color: vec3<f32>,
    environment_rotation: f32,
}
@group(0) @binding(7) var<uniform> camera: CameraUniform;

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
@group(0) @binding(8) var<storage, read> materials: array<MaterialData3D>;
@group(0) @binding(9) var environment_map: texture_2d<f32>;
@group(0) @binding(10) var environment_sampler: sampler;

struct SurfaceRecord {
    normal: vec3<f32>,
    linear_depth: f32,
    albedo_alpha: vec4<f32>,
    material_index: u32,
    coverage: f32,
    valid: u32,
}

fn reconstruct_position(pixel: vec2<u32>, linear_depth: f32) -> vec3<f32> {
    let forward = normalize(camera.look_at);
    let right = normalize(cross(forward, camera.up));
    let up = cross(right, forward);
    let dimensions = vec2<f32>(textureDimensions(output_tex));
    let ndc_x = ((f32(pixel.x) + 0.5) / dimensions.x) * 2.0 - 1.0;
    let ndc_y = 1.0 - ((f32(pixel.y) + 0.5) / dimensions.y) * 2.0;

    if (camera.proj_type == 0u) {
        let aspect = dimensions.x / dimensions.y;
        let direction = normalize(
            right * ndc_x * aspect * tan(camera.fov * 0.5)
                + up * ndc_y * tan(camera.fov * 0.5)
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

fn load_sdf_record(pixel: vec2<u32>, output_dimensions: vec2<u32>) -> SurfaceRecord {
    let sdf_dimensions = textureDimensions(sdf_normal_coverage_tex);
    let sdf_pixel = min(pixel * sdf_dimensions / output_dimensions, sdf_dimensions - vec2<u32>(1));
    let position = vec2<i32>(sdf_pixel);
    let normal_coverage = textureLoad(sdf_normal_coverage_tex, position, 0);
    if (normal_coverage.a <= 0.0) {
        return SurfaceRecord(vec3<f32>(0.0), 1e20, vec4<f32>(0.0), 0u, 0.0, 0u);
    }
    let material_index = textureLoad(sdf_material_id_tex, position, 0).x;
    return SurfaceRecord(
        normalize(normal_coverage.xyz),
        textureLoad(sdf_depth_tex, position, 0).x,
        materials[material_index].base_color,
        material_index,
        normal_coverage.a,
        1u,
    );
}

fn load_raster_record(pixel: vec2<u32>) -> SurfaceRecord {
    var selected_normal_depth = vec4<f32>(0.0, 0.0, 0.0, 1e20);
    var selected_albedo = vec4<f32>(0.0);
    var selected_material = 0u;
    var covered_samples = 0u;
    for (var sample = 0u; sample < GBUFFER_SAMPLE_COUNT; sample += 1u) {
        let normal_depth = load_raster_normal_depth(vec2<i32>(pixel), sample);
        if (dot(normal_depth.xyz, normal_depth.xyz) <= 0.25) {
            continue;
        }
        covered_samples += 1u;
        if (normal_depth.w < selected_normal_depth.w) {
            selected_normal_depth = normal_depth;
            selected_albedo = load_raster_albedo(vec2<i32>(pixel), sample);
            selected_material = load_raster_material_id(vec2<i32>(pixel), sample);
        }
    }
    if (covered_samples == 0u) {
        return SurfaceRecord(vec3<f32>(0.0), 1e20, vec4<f32>(0.0), 0u, 0.0, 0u);
    }
    return SurfaceRecord(
        normalize(selected_normal_depth.xyz),
        selected_normal_depth.w,
        selected_albedo,
        selected_material,
        f32(covered_samples) / f32(GBUFFER_SAMPLE_COUNT),
        1u,
    );
}

fn shade_record(record: SurfaceRecord, coverage: f32, pixel: vec2<u32>) -> vec4<f32> {
    if (record.valid == 0u || coverage <= 0.0) {
        return vec4<f32>(0.0);
    }
    let material = materials[record.material_index];
    let position = reconstruct_position(pixel, record.linear_depth);
    let albedo = record.albedo_alpha.rgb;
    let roughness = clamp(material.surface.x, 0.04, 1.0);
    let metallic = clamp(material.surface.y, 0.0, 1.0);
    let reflectance_f0 = 0.16 * material.surface.z * material.surface.z;
    let f0 = mix(vec3<f32>(reflectance_f0), albedo, metallic);
    let lighting = shade_surface(
        position,
        record.normal,
        normalize(camera.pos - position),
        albedo,
        roughness,
        metallic,
        f0,
        material.emissive.rgb * material.emissive.a,
    );
    let alpha = record.albedo_alpha.a * coverage;
    return vec4<f32>(lighting.color * alpha, alpha);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dimensions = textureDimensions(output_tex);
    if (global_id.x >= dimensions.x || global_id.y >= dimensions.y) {
        return;
    }

    var sdf = SurfaceRecord(vec3<f32>(0.0), 1e20, vec4<f32>(0.0), 0u, 0.0, 0u);
    var raster = SurfaceRecord(vec3<f32>(0.0), 1e20, vec4<f32>(0.0), 0u, 0.0, 0u);
    if (camera.num_primitives != 0u) {
        sdf = load_sdf_record(global_id.xy, dimensions);
    }
    if (camera.has_raster_surfaces != 0u) {
        raster = load_raster_record(global_id.xy);
    }
    var color = vec4<f32>(0.0);
    if (sdf.valid != 0u && raster.valid != 0u) {
        if (raster.linear_depth < sdf.linear_depth) {
            color = shade_record(raster, raster.coverage, global_id.xy);
            color += shade_record(sdf, sdf.coverage * (1.0 - raster.coverage), global_id.xy);
        } else {
            color = shade_record(sdf, sdf.coverage, global_id.xy);
            color += shade_record(raster, raster.coverage * (1.0 - sdf.coverage), global_id.xy);
        }
    } else if (sdf.valid != 0u) {
        color = shade_record(sdf, sdf.coverage, global_id.xy);
    } else if (raster.valid != 0u) {
        color = shade_record(raster, raster.coverage, global_id.xy);
    }
    textureStore(output_tex, vec2<i32>(global_id.xy), color);
}
