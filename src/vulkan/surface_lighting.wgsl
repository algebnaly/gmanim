struct SurfaceLighting {
    color: vec3<f32>,
    environment_fresnel: vec3<f32>,
}

fn distribution_ggx(normal: vec3<f32>, half_direction: vec3<f32>, roughness: f32) -> f32 {
    let alpha = roughness * roughness;
    let alpha_squared = alpha * alpha;
    let n_dot_h = max(dot(normal, half_direction), 0.0);
    let denominator = n_dot_h * n_dot_h * (alpha_squared - 1.0) + 1.0;
    return alpha_squared / max(3.14159265359 * denominator * denominator, 1e-5);
}

fn geometry_schlick_ggx(n_dot_direction: f32, roughness: f32) -> f32 {
    let remapped = roughness + 1.0;
    let k = remapped * remapped / 8.0;
    return n_dot_direction / max(n_dot_direction * (1.0 - k) + k, 1e-5);
}

fn geometry_smith(
    normal: vec3<f32>,
    view_direction: vec3<f32>,
    light_direction: vec3<f32>,
    roughness: f32,
) -> f32 {
    return geometry_schlick_ggx(max(dot(normal, view_direction), 0.0), roughness)
        * geometry_schlick_ggx(max(dot(normal, light_direction), 0.0), roughness);
}

fn fresnel_schlick(cosine: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - clamp(cosine, 0.0, 1.0), 5.0);
}

fn sample_environment(direction: vec3<f32>, roughness: f32) -> vec3<f32> {
    let cosine = cos(camera.environment_rotation);
    let sine = sin(camera.environment_rotation);
    let rotated = vec3<f32>(
        direction.x * cosine - direction.z * sine,
        direction.y,
        direction.x * sine + direction.z * cosine,
    );
    let uv = vec2<f32>(
        atan2(rotated.z, rotated.x) / 6.28318530718 + 0.5,
        acos(clamp(rotated.y, -1.0, 1.0)) / 3.14159265359,
    );
    return textureSampleLevel(
        environment_map,
        environment_sampler,
        uv,
        clamp(roughness, 0.0, 1.0) * 8.0,
    ).rgb;
}

fn environment_brdf(f0: vec3<f32>, roughness: f32, n_dot_v: f32) -> vec3<f32> {
    let c0 = vec4<f32>(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4<f32>(1.0, 0.0425, 1.04, -0.04);
    let r = roughness * c0 + c1;
    let a004 = min(r.x * r.x, exp2(-9.28 * n_dot_v)) * r.x + r.y;
    let scale_bias = vec2<f32>(-1.04, 1.04) * a004 + r.zw;
    return f0 * scale_bias.x + scale_bias.y;
}

fn shade_surface(
    position: vec3<f32>,
    normal: vec3<f32>,
    view_direction: vec3<f32>,
    albedo: vec3<f32>,
    roughness: f32,
    metallic: f32,
    f0: vec3<f32>,
    emissive: vec3<f32>,
) -> SurfaceLighting {
    let light_direction = normalize(camera.light_pos - position);
    let half_direction = normalize(light_direction + view_direction);
    let n_dot_l = max(dot(normal, light_direction), 0.0);
    let n_dot_v = max(dot(normal, view_direction), 0.0);
    let h_dot_v = max(dot(half_direction, view_direction), 0.0);
    let fresnel = fresnel_schlick(h_dot_v, f0);
    let specular_brdf = distribution_ggx(normal, half_direction, roughness)
        * geometry_smith(normal, view_direction, light_direction, roughness)
        * fresnel
        / max(4.0 * n_dot_v * n_dot_l, 1e-4);
    let diffuse_brdf = (vec3<f32>(1.0) - fresnel)
        * (1.0 - metallic)
        * albedo
        / 3.14159265359;
    let light_delta = camera.light_pos - position;
    let radiance = camera.light_color
        * camera.light_intensity
        / (12.5663706144 * max(dot(light_delta, light_delta), 1e-3));
    var color = (diffuse_brdf + specular_brdf) * radiance * n_dot_l;

    let environment_tint = camera.environment_color * camera.environment_intensity;
    let environment_fresnel = fresnel_schlick(n_dot_v, f0);
    color += (vec3<f32>(1.0) - environment_fresnel)
        * (1.0 - metallic)
        * albedo
        * sample_environment(normal, 1.0)
        * environment_tint;
    color += environment_brdf(f0, roughness, n_dot_v)
        * sample_environment(reflect(-view_direction, normal), roughness)
        * environment_tint;
    return SurfaceLighting(color + emissive, environment_fresnel);
}
