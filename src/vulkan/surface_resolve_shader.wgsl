@group(0) @binding(0) var resolved_primary_normal_depth: texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var resolved_primary_albedo_coverage: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var resolved_secondary_normal_depth: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var resolved_secondary_albedo_coverage: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var resolved_material_ids: texture_storage_2d<r32uint, write>;

@group(0) @binding(11) var<uniform> camera: CameraUniform;
@group(0) @binding(12) var<storage, read> materials: array<MaterialData3D>;

struct SurfaceRecord {
    normal: vec3<f32>,
    linear_depth: f32,
    albedo_alpha: vec4<f32>,
    material_index: u32,
    coverage: f32,
    valid: u32,
}

fn empty_surface() -> SurfaceRecord {
    return SurfaceRecord(vec3<f32>(0.0), 0.0, vec4<f32>(0.0), 0u, 0.0, 0u);
}

fn load_sdf_record(pixel: vec2<u32>, output_dimensions: vec2<u32>) -> SurfaceRecord {
    let sdf_dimensions = textureDimensions(sdf_normal_coverage_tex);
    let sdf_pixel = min(pixel * sdf_dimensions / output_dimensions, sdf_dimensions - vec2<u32>(1));
    let position = vec2<i32>(sdf_pixel);
    let normal_coverage = textureLoad(sdf_normal_coverage_tex, position, 0);
    if (normal_coverage.a <= 0.0) {
        return empty_surface();
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
    var selected_normal_depth = vec4<f32>(0.0);
    var selected_albedo = vec4<f32>(0.0);
    var selected_material = 0u;
    var covered_samples = 0u;
    var nearest_depth = 1e20;
    for (var sample = 0u; sample < GBUFFER_SAMPLE_COUNT; sample += 1u) {
        let normal_depth = load_raster_normal_depth(vec2<i32>(pixel), sample);
        if (dot(normal_depth.xyz, normal_depth.xyz) <= 0.25) {
            continue;
        }
        covered_samples += 1u;
        if (normal_depth.w < nearest_depth) {
            nearest_depth = normal_depth.w;
            selected_normal_depth = normal_depth;
            selected_albedo = load_raster_albedo(vec2<i32>(pixel), sample);
            selected_material = load_raster_material_id(vec2<i32>(pixel), sample);
        }
    }
    if (covered_samples == 0u) {
        return empty_surface();
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

fn encoded_normal_depth(surface: SurfaceRecord) -> vec4<f32> {
    if (surface.valid == 0u) {
        return vec4<f32>(0.0);
    }
    return vec4<f32>(surface.normal, surface.linear_depth);
}

fn encoded_albedo_coverage(surface: SurfaceRecord, coverage: f32) -> vec4<f32> {
    if (surface.valid == 0u) {
        return vec4<f32>(0.0);
    }
    return vec4<f32>(surface.albedo_alpha.rgb, surface.albedo_alpha.a * coverage);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dimensions = textureDimensions(resolved_primary_normal_depth);
    if (any(global_id.xy >= dimensions)) {
        return;
    }

    var sdf = empty_surface();
    var raster = empty_surface();
    if (camera.num_primitives != 0u) {
        sdf = load_sdf_record(global_id.xy, dimensions);
    }
    if (camera.has_raster_surfaces != 0u) {
        raster = load_raster_record(global_id.xy);
    }

    var primary = empty_surface();
    var secondary = empty_surface();
    var primary_coverage = 0.0;
    var secondary_coverage = 0.0;
    if (sdf.valid != 0u && raster.valid != 0u) {
        if (raster.linear_depth < sdf.linear_depth) {
            primary = raster;
            secondary = sdf;
        } else {
            primary = sdf;
            secondary = raster;
        }
        primary_coverage = primary.coverage;
        secondary_coverage = secondary.coverage * (1.0 - primary.coverage);
    } else if (sdf.valid != 0u) {
        primary = sdf;
        primary_coverage = sdf.coverage;
    } else if (raster.valid != 0u) {
        primary = raster;
        primary_coverage = raster.coverage;
    }

    let position = vec2<i32>(global_id.xy);
    textureStore(resolved_primary_normal_depth, position, encoded_normal_depth(primary));
    textureStore(
        resolved_primary_albedo_coverage,
        position,
        encoded_albedo_coverage(primary, primary_coverage),
    );
    textureStore(resolved_secondary_normal_depth, position, encoded_normal_depth(secondary));
    textureStore(
        resolved_secondary_albedo_coverage,
        position,
        encoded_albedo_coverage(secondary, secondary_coverage),
    );
    textureStore(
        resolved_material_ids,
        position,
        vec4<u32>(
            (primary.material_index & 0xffffu) | (secondary.material_index << 16u),
            0u,
            0u,
            0u,
        ),
    );
}
