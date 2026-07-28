@group(0) @binding(0) var output_hdr: texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var resolved_primary_normal_depth: texture_2d<f32>;
@group(0) @binding(2) var resolved_primary_albedo_coverage: texture_2d<f32>;
@group(0) @binding(3) var resolved_secondary_normal_depth: texture_2d<f32>;
@group(0) @binding(4) var resolved_secondary_albedo_coverage: texture_2d<f32>;
@group(0) @binding(5) var resolved_material_ids: texture_2d<u32>;
@group(0) @binding(6) var<uniform> camera: CameraUniform;
@group(0) @binding(7) var<storage, read> materials: array<MaterialData3D>;
@group(0) @binding(8) var environment_map: texture_2d<f32>;
@group(0) @binding(9) var environment_sampler: sampler;

fn reconstruct_position(pixel: vec2<u32>, linear_depth: f32) -> vec3<f32> {
    let forward = normalize(camera.look_at);
    let right = normalize(cross(forward, camera.up));
    let up = cross(right, forward);
    let dimensions = vec2<f32>(textureDimensions(output_hdr));
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

fn shade_resolved_surface(
    pixel: vec2<u32>,
    normal_depth: vec4<f32>,
    albedo_coverage: vec4<f32>,
    material_index: u32,
) -> vec4<f32> {
    if (albedo_coverage.a <= 0.0 || dot(normal_depth.xyz, normal_depth.xyz) <= 0.25) {
        return vec4<f32>(0.0);
    }
    let material = materials[material_index];
    let position = reconstruct_position(pixel, normal_depth.w);
    let normal = normalize(normal_depth.xyz);
    let albedo = albedo_coverage.rgb;
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
    return vec4<f32>(lighting.color * albedo_coverage.a, albedo_coverage.a);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dimensions = textureDimensions(output_hdr);
    if (any(global_id.xy >= dimensions)) {
        return;
    }
    let position = vec2<i32>(global_id.xy);
    let packed_material_ids = textureLoad(resolved_material_ids, position, 0).x;
    let primary_material_id = packed_material_ids & 0xffffu;
    let secondary_material_id = packed_material_ids >> 16u;
    let primary = shade_resolved_surface(
        global_id.xy,
        textureLoad(resolved_primary_normal_depth, position, 0),
        textureLoad(resolved_primary_albedo_coverage, position, 0),
        primary_material_id,
    );
    let secondary = shade_resolved_surface(
        global_id.xy,
        textureLoad(resolved_secondary_normal_depth, position, 0),
        textureLoad(resolved_secondary_albedo_coverage, position, 0),
        secondary_material_id,
    );
    textureStore(
        output_hdr,
        position,
        vec4<f32>(primary.rgb + secondary.rgb, min(primary.a + secondary.a, 1.0)),
    );
}
