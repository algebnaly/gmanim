@group(0) @binding(0) var input_img: texture_2d<f32>;
@group(0) @binding(1) var y_plane: texture_storage_2d<r8unorm, write>;
@group(0) @binding(2) var uv_plane: texture_storage_2d<rg8unorm, write>;

fn aces_tone_map(color: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp(
        (color * (a * color + vec3<f32>(b)))
            / (color * (c * color + vec3<f32>(d)) + vec3<f32>(e)),
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
}

fn rgb_to_yuv(rgb: vec3<f32>) -> vec3<f32> {
    let y = 16.0 / 255.0 + 0.256788 * rgb.r + 0.504129 * rgb.g + 0.097906 * rgb.b;
    let u = 128.0 / 255.0 - 0.148223 * rgb.r - 0.290993 * rgb.g + 0.439216 * rgb.b;
    let v = 128.0 / 255.0 + 0.439216 * rgb.r - 0.367788 * rgb.g - 0.071427 * rgb.b;
    return vec3<f32>(y, u, v);
}

fn average_2x2(base: vec2<u32>) -> vec3<f32> {
    let color = (
        textureLoad(input_img, vec2<i32>(base), 0).rgb
        + textureLoad(input_img, vec2<i32>(base + vec2<u32>(1u, 0u)), 0).rgb
        + textureLoad(input_img, vec2<i32>(base + vec2<u32>(0u, 1u)), 0).rgb
        + textureLoad(input_img, vec2<i32>(base + vec2<u32>(1u, 1u)), 0).rgb
    ) * 0.25;
    return aces_tone_map(color);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let output_size = textureDimensions(y_plane);
    let uv_size = output_size / 2u;
    if (id.x >= uv_size.x || id.y >= uv_size.y) {
        return;
    }

    let input_base = id.xy * 4u;
    let output_base = id.xy * 2u;
    let rgb00 = average_2x2(input_base);
    let rgb10 = average_2x2(input_base + vec2<u32>(2u, 0u));
    let rgb01 = average_2x2(input_base + vec2<u32>(0u, 2u));
    let rgb11 = average_2x2(input_base + vec2<u32>(2u, 2u));
    let yuv00 = rgb_to_yuv(rgb00);
    let yuv10 = rgb_to_yuv(rgb10);
    let yuv01 = rgb_to_yuv(rgb01);
    let yuv11 = rgb_to_yuv(rgb11);

    textureStore(y_plane, vec2<i32>(output_base), vec4<f32>(yuv00.x, 0.0, 0.0, 1.0));
    textureStore(
        y_plane,
        vec2<i32>(output_base + vec2<u32>(1u, 0u)),
        vec4<f32>(yuv10.x, 0.0, 0.0, 1.0),
    );
    textureStore(
        y_plane,
        vec2<i32>(output_base + vec2<u32>(0u, 1u)),
        vec4<f32>(yuv01.x, 0.0, 0.0, 1.0),
    );
    textureStore(
        y_plane,
        vec2<i32>(output_base + vec2<u32>(1u, 1u)),
        vec4<f32>(yuv11.x, 0.0, 0.0, 1.0),
    );

    let uv = (yuv00.yz + yuv10.yz + yuv01.yz + yuv11.yz) * 0.25;
    textureStore(uv_plane, vec2<i32>(id.xy), vec4<f32>(uv, 0.0, 1.0));
}
