@group(0) @binding(0) var input_image: texture_2d<f32>;
@group(0) @binding(1) var output_image: texture_storage_2d<rgba16float, write>;

fn load_clamped(coord: vec2<i32>) -> vec3<f32> {
    let size = vec2<i32>(textureDimensions(input_image));
    return textureLoad(input_image, clamp(coord, vec2<i32>(0), size - vec2<i32>(1)), 0).rgb;
}

@compute @workgroup_size(16, 16, 1)
fn extract(@builtin(global_invocation_id) id: vec3<u32>) {
    let output_size = textureDimensions(output_image);
    if id.x >= output_size.x || id.y >= output_size.y {
        return;
    }

    let input_size = textureDimensions(input_image);
    let base = vec2<i32>(id.xy * input_size / output_size);
    let color = (
        load_clamped(base)
        + load_clamped(base + vec2<i32>(1, 0))
        + load_clamped(base + vec2<i32>(0, 1))
        + load_clamped(base + vec2<i32>(1, 1))
    ) * 0.25;
    let brightness = max(color.r, max(color.g, color.b));
    let contribution = max(brightness - 1.0, 0.0) / max(brightness, 1e-4);
    textureStore(output_image, vec2<i32>(id.xy), vec4<f32>(color * contribution, 1.0));
}

fn gaussian_blur(coord: vec2<i32>, direction: vec2<i32>) -> vec3<f32> {
    var color = load_clamped(coord) * 0.22702703;
    color += load_clamped(coord + direction) * 0.19459459;
    color += load_clamped(coord - direction) * 0.19459459;
    color += load_clamped(coord + direction * 2) * 0.12162162;
    color += load_clamped(coord - direction * 2) * 0.12162162;
    color += load_clamped(coord + direction * 3) * 0.05405405;
    color += load_clamped(coord - direction * 3) * 0.05405405;
    color += load_clamped(coord + direction * 4) * 0.01621622;
    color += load_clamped(coord - direction * 4) * 0.01621622;
    return color;
}

@compute @workgroup_size(16, 16, 1)
fn blur_horizontal(@builtin(global_invocation_id) id: vec3<u32>) {
    let output_size = textureDimensions(output_image);
    if id.x >= output_size.x || id.y >= output_size.y {
        return;
    }
    let color = gaussian_blur(vec2<i32>(id.xy), vec2<i32>(1, 0));
    textureStore(output_image, vec2<i32>(id.xy), vec4<f32>(color, 1.0));
}

@compute @workgroup_size(16, 16, 1)
fn blur_vertical(@builtin(global_invocation_id) id: vec3<u32>) {
    let output_size = textureDimensions(output_image);
    if id.x >= output_size.x || id.y >= output_size.y {
        return;
    }
    let color = gaussian_blur(vec2<i32>(id.xy), vec2<i32>(0, 1));
    textureStore(output_image, vec2<i32>(id.xy), vec4<f32>(color, 1.0));
}
