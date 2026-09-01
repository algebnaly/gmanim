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

@group(0) @binding(1) var<uniform> camera: CameraUniform;
@group(0) @binding(2) var<storage, read> materials: array<MaterialData3D>;
@group(0) @binding(3) var scene_color: texture_2d<f32>;
@group(0) @binding(4) var transparent_back_depth: texture_2d<f32>;
@group(0) @binding(5) var environment_map: texture_2d<f32>;
@group(0) @binding(6) var environment_sampler: sampler;
@group(0) @binding(7) var sdf_depth: texture_2d<f32>;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) surface_coord: vec3<f32>,
    @builtin(instance_index) material_index: u32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) frag_pos: vec3<f32>,
    @location(3) surface_coord: vec3<f32>,
    @location(4) @interpolate(flat) material_index: u32,
}

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let w = -normalize(camera.look_at);
    let u = normalize(cross(camera.up, w));
    let v = cross(w, u);
    let view_mat = mat4x4<f32>(
        vec4<f32>(u.x, v.x, w.x, 0.0),
        vec4<f32>(u.y, v.y, w.y, 0.0),
        vec4<f32>(u.z, v.z, w.z, 0.0),
        vec4<f32>(-dot(u, camera.pos), -dot(v, camera.pos), -dot(w, camera.pos), 1.0)
    );

    let world_pos = vec4<f32>(model.position, 1.0);
    out.frag_pos = world_pos.xyz;
    out.clip_position = camera.proj_mat * view_mat * world_pos;
    out.normal = model.normal;
    out.color = model.color;
    out.surface_coord = model.surface_coord;
    out.material_index = model.material_index;
    return out;
}

fn raster_pixel_scale() -> f32 {
    return f32(max(camera.raster_scale, 1u));
}

fn antialiased_periodic_line(coord: f32, width_pixels: f32) -> f32 {
    if width_pixels < 0.02 {
        return 0.0;
    }
    let grad = vec2<f32>(dpdx(coord), dpdy(coord));
    let deriv = max(length(grad), 1e-6);
    let dist_coord = abs(fract(coord + 0.5) - 0.5);
    let dist_pixels = dist_coord / deriv;

    // Derivatives are measured in raster pixels. SSAA makes those pixels
    // smaller, so convert output-pixel widths to raster pixels before
    // evaluating coverage. Conversely, the Nyquist decision describes the
    // final output footprint and must convert the derivative back to output
    // pixels. Without both conversions, SSAA changes line width and which
    // distant grid frequencies remain visible.
    let scale = raster_pixel_scale();
    let target_w = max(width_pixels * scale, 0.01);
    let draw_w = max(target_w, 1.0);
    let coverage = clamp(draw_w * 0.5 + 0.5 - dist_pixels, 0.0, 1.0);
    let subpixel_alpha = min(target_w, 1.0);
    let output_deriv = deriv * scale;
    let nyquist_fade = clamp(1.0 - (output_deriv - 0.2) / 0.25, 0.0, 1.0);

    return coverage * subpixel_alpha * nyquist_fade;
}

fn spherical_grid(surface_coord: vec3<f32>, material: MaterialData3D) -> f32 {
    let n = normalize(surface_coord);
    let longitude = atan2(n.z, n.x) * (material.grid.x / 6.28318530718);
    let latitude = (asin(clamp(n.y, -1.0, 1.0)) / 3.14159265359 + 0.5) * material.grid.y;
    let longitude_line = antialiased_periodic_line(longitude, material.grid.z);
    let latitude_line = antialiased_periodic_line(latitude, material.grid.z);
    return max(longitude_line, latitude_line) * step(0.5, material.grid.w);
}

fn antialiased_line(distance: f32, width_pixels: f32) -> f32 {
    if width_pixels < 0.02 {
        return 0.0;
    }
    let grad = vec2<f32>(dpdx(distance), dpdy(distance));
    let deriv = max(length(grad), 1e-6);
    let dist_pixels = abs(distance) / deriv;
    
    let target_w = max(width_pixels * raster_pixel_scale(), 0.01);
    let draw_w = max(target_w, 1.0);
    let coverage = clamp(draw_w * 0.5 + 0.5 - dist_pixels, 0.0, 1.0);
    let subpixel_alpha = min(target_w, 1.0);
    
    return coverage * subpixel_alpha;
}

struct SphericalPatchMasks {
    fill: f32,
    edge: f32,
}

fn spherical_patch(
    surface_coord: vec3<f32>,
    material: MaterialData3D,
) -> SphericalPatchMasks {
    if material.patch_color.a <= 1e-5 {
        return SphericalPatchMasks(0.0, 0.0);
    }
    let point = normalize(surface_coord);
    let a = normalize(material.patch_corner_0.xyz);
    let b = normalize(material.patch_corner_1.xyz);
    let c = normalize(material.patch_corner_2.xyz);
    let interior = normalize(a + b + c);

    let edge_ab = cross(a, b);
    let edge_bc = cross(b, c);
    let edge_ca = cross(c, a);
    let oriented_ab = normalize(select(-edge_ab, edge_ab, dot(edge_ab, interior) >= 0.0));
    let oriented_bc = normalize(select(-edge_bc, edge_bc, dot(edge_bc, interior) >= 0.0));
    let oriented_ca = normalize(select(-edge_ca, edge_ca, dot(edge_ca, interior) >= 0.0));
    let distance_ab = dot(point, oriented_ab);
    let distance_bc = dot(point, oriented_bc);
    let distance_ca = dot(point, oriented_ca);
    let distance = min(distance_ab, min(distance_bc, distance_ca));
    let antialias = max(fwidth(distance), 1e-5);
    let fill = smoothstep(-antialias, antialias, distance);

    let edge_width = material.patch_params.x;
    let gate_width = max(
        max(fwidth(distance_ab), fwidth(distance_bc)),
        fwidth(distance_ca),
    ) * max(edge_width * raster_pixel_scale(), 1.0);
    let edge_ab_mask = antialiased_line(distance_ab, edge_width)
        * step(-gate_width, distance_bc)
        * step(-gate_width, distance_ca);
    let edge_bc_mask = antialiased_line(distance_bc, edge_width)
        * step(-gate_width, distance_ca)
        * step(-gate_width, distance_ab);
    let edge_ca_mask = antialiased_line(distance_ca, edge_width)
        * step(-gate_width, distance_ab)
        * step(-gate_width, distance_bc);
    return SphericalPatchMasks(fill, max(edge_ab_mask, max(edge_bc_mask, edge_ca_mask)));
}

fn world_to_clip(world_position: vec3<f32>) -> vec4<f32> {
    let w = -normalize(camera.look_at);
    let u = normalize(cross(camera.up, w));
    let v = cross(w, u);
    let view_position = vec4<f32>(
        dot(u, world_position - camera.pos),
        dot(v, world_position - camera.pos),
        dot(w, world_position - camera.pos),
        1.0,
    );
    return camera.proj_mat * view_position;
}

fn refracted_scene_color(
    fragment_position: vec2<f32>,
    world_position: vec3<f32>,
    geometric_normal: vec3<f32>,
    view_direction: vec3<f32>,
    optical_path: f32,
    ior: f32,
) -> vec3<f32> {
    let incident = -view_direction;
    let entering_medium = dot(incident, geometric_normal) < 0.0;
    let interface_normal = select(-geometric_normal, geometric_normal, entering_medium);
    let eta = select(ior, 1.0 / ior, entering_medium);
    var refracted_direction = refract(incident, interface_normal, eta);
    if dot(refracted_direction, refracted_direction) < 1e-5 {
        refracted_direction = reflect(incident, interface_normal);
    }

    let start_clip = world_to_clip(world_position);
    let end_clip = world_to_clip(world_position + refracted_direction * optical_path);
    let start_ndc = start_clip.xy / max(abs(start_clip.w), 1e-5);
    let end_ndc = end_clip.xy / max(abs(end_clip.w), 1e-5);
    let dimensions = textureDimensions(scene_color);
    let pixel_offset = (end_ndc - start_ndc) * vec2<f32>(dimensions) * 0.5;
    let maximum = vec2<i32>(dimensions) - vec2<i32>(1);
    let sample_position = clamp(
        vec2<i32>(fragment_position + pixel_offset),
        vec2<i32>(0),
        maximum,
    );
    return textureLoad(scene_color, sample_position, 0).rgb;
}

@fragment
fn fs_back_depth(in: VertexOutput) -> @location(0) f32 {
    let camera_forward = normalize(camera.look_at);
    return max(dot(in.frag_pos - camera.pos, camera_forward), 0.0);
}

struct GBufferOutput {
    @location(0) normal_depth: vec4<f32>,
    @location(1) albedo: vec4<f32>,
    @location(2) material_index: u32,
}

@fragment
fn fs_gbuffer(in: VertexOutput, @builtin(front_facing) is_front: bool) -> GBufferOutput {
    let material = materials[in.material_index];
    let geometric_normal = normalize(in.normal);
    let normal = select(-geometric_normal, geometric_normal, is_front);
    var albedo = in.color.rgb * material.base_color.rgb;

    let patch_masks = spherical_patch(in.surface_coord, material);
    albedo = mix(
        albedo,
        material.patch_color.rgb,
        patch_masks.fill * material.patch_color.a,
    );
    let grid_mask = spherical_grid(in.surface_coord, material);
    let face_intensity = select(material.grid_backface.x, 1.0, is_front);
    let grid_mix = clamp(grid_mask * material.grid_color.a * face_intensity, 0.0, 1.0);
    albedo = mix(albedo, material.grid_color.rgb, grid_mix);
    albedo = mix(
        albedo,
        material.patch_edge_color.rgb,
        patch_masks.edge * material.patch_edge_color.a,
    );

    let alpha = max(
        in.color.a * material.base_color.a,
        max(
            grid_mask * material.grid_color.a * face_intensity,
            max(
                patch_masks.fill * material.patch_color.a,
                patch_masks.edge * material.patch_edge_color.a,
            ),
        ),
    );
    let linear_depth = dot(in.frag_pos - camera.pos, normalize(camera.look_at));
    return GBufferOutput(
        vec4<f32>(normal, linear_depth),
        vec4<f32>(albedo, alpha),
        in.material_index,
    );
}

@fragment
fn fs_main(in: VertexOutput, @builtin(front_facing) is_front: bool) -> @location(0) vec4<f32> {
    if camera.num_primitives > 0u {
        // sdf_depth is allocated at the raster resolution; map with a ratio
        // so the lookup stays correct if the resolutions ever diverge.
        let depth_dimensions = textureDimensions(sdf_depth);
        let scale = f32(max(camera.raster_scale, 1u));
        let raster_size = vec2<u32>(
            max(u32(camera.width * scale), 1u),
            max(u32(camera.height * scale), 1u),
        );
        let depth_position = min(
            vec2<u32>(in.clip_position.xy) * depth_dimensions / raster_size,
            depth_dimensions - vec2<u32>(1),
        );
        let sdf_linear_depth = textureLoad(sdf_depth, vec2<i32>(depth_position), 0).r;
        let mesh_linear_depth = dot(
            in.frag_pos - camera.pos,
            normalize(camera.look_at),
        );
        if sdf_linear_depth + 1e-4 < mesh_linear_depth {
            discard;
        }
    }

    let material = materials[in.material_index];
    let geometric_normal = normalize(in.normal);
    let normal = select(-geometric_normal, geometric_normal, is_front);
    let view_direction = normalize(camera.pos - in.frag_pos);

    let is_transparent = material.surface.w >= 0.5;
    let albedo = in.color.rgb * material.base_color.rgb;
    let is_unlit = material.grid_backface.y > 0.5;
    let is_flat = material.grid_backface.z > 0.5;
    var color = vec3<f32>(0.0);
    let emissive_color = material.emissive.rgb * material.emissive.a;
    var environment_fresnel = vec3<f32>(1.0);

    if is_unlit {
        color = albedo + emissive_color;
    } else if is_flat {
        let lighting = shade_surface_flat(
            in.frag_pos,
            normal,
            view_direction,
            albedo,
            emissive_color,
        );
        color = lighting.color;
        environment_fresnel = lighting.environment_fresnel;
    } else {
        let roughness = clamp(material.surface.x, 0.04, 1.0);
        let metallic = clamp(material.surface.y, 0.0, 1.0);
        let reflectance_f0 = 0.16 * material.surface.z * material.surface.z;
        let ior_f0 = pow((material.transmission.z - 1.0) / (material.transmission.z + 1.0), 2.0);
        let dielectric_f0 = select(reflectance_f0, ior_f0, is_transparent);
        let f0 = mix(vec3<f32>(dielectric_f0), albedo, metallic);
        let lighting = shade_surface(
            in.frag_pos,
            normal,
            view_direction,
            albedo,
            roughness,
            metallic,
            f0,
            emissive_color,
        );
        color = lighting.color;
        environment_fresnel = lighting.environment_fresnel;
    }

    var optical_path = 0.0;
    if is_transparent {
        let depth_sample_position = clamp(
            vec2<i32>(in.clip_position.xy),
            vec2<i32>(0),
            vec2<i32>(textureDimensions(transparent_back_depth)) - vec2<i32>(1),
        );
        let back_depth = textureLoad(transparent_back_depth, depth_sample_position, 0).r;
        let front_depth = dot(in.frag_pos - camera.pos, normalize(camera.look_at));
        optical_path = select(0.0, max(back_depth - front_depth, 0.0), is_front);
    }
    let transmittance = exp(-material.absorption.rgb * optical_path);
    let absorption_alpha = 1.0
        - dot(transmittance, vec3<f32>(0.2126, 0.7152, 0.0722));
    let medium_scattering = albedo
        * (absorption_alpha * 0.9 + material.transmission.x * 0.25);
    if is_transparent {
        let refracted = refracted_scene_color(
            in.clip_position.xy,
            in.frag_pos,
            geometric_normal,
            view_direction,
            optical_path,
            max(material.transmission.z, 1.0001),
        );
        let transmitted = refracted * transmittance + medium_scattering;
        color = mix(transmitted, color, environment_fresnel);
    }

    let patch_masks = spherical_patch(in.surface_coord, material);
    color = mix(
        color,
        material.patch_color.rgb,
        patch_masks.fill * material.patch_color.a,
    );

    let grid_mask = spherical_grid(in.surface_coord, material);
    let face_intensity = select(material.grid_backface.x, 1.0, is_front);
    let grid_mix = clamp(grid_mask * material.grid_color.a * face_intensity, 0.0, 1.0);
    color = mix(color, material.grid_color.rgb, grid_mix);
    color = mix(
        color,
        material.patch_edge_color.rgb,
        patch_masks.edge * material.patch_edge_color.a,
    );

    let base_medium_alpha = 1.0
        - (1.0 - material.transmission.x) * (1.0 - absorption_alpha);
    let n_dot_v = max(dot(normal, view_direction), 0.0);
    let edge_fresnel = pow(1.0 - n_dot_v, 5.0) * material.transmission.y;
    let medium_alpha = 1.0 - (1.0 - base_medium_alpha) * (1.0 - edge_fresnel);
    let face_opacity = select(material.absorption.w, 1.0, is_front);
    let opaque_alpha = in.color.a * material.base_color.a;
    let transparent_alpha = opaque_alpha * medium_alpha * face_opacity;
    let surface_alpha = clamp(
        select(opaque_alpha, transparent_alpha, is_transparent),
        0.0,
        1.0,
    );

    let grid_alpha = grid_mask * material.grid_color.a * face_intensity;
    let patch_alpha = patch_masks.fill * material.patch_color.a;
    let patch_edge_alpha = patch_masks.edge * material.patch_edge_color.a;
    let alpha = max(
        surface_alpha,
        max(grid_alpha, max(patch_alpha, patch_edge_alpha)),
    );
    return vec4<f32>(color, alpha);
}
