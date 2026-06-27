@group(0) @binding(0) var input_img: texture_storage_2d<rgba8unorm, read>;
@group(0) @binding(1) var y_plane: texture_storage_2d<r8unorm, write>;
@group(0) @binding(2) var uv_plane: texture_storage_2d<rg8unorm, write>;

fn rgb_to_yuv(rgb: vec3<f32>) -> vec3<f32> {
    let y = 16.0 / 255.0 + 0.256788 * rgb.r + 0.504129 * rgb.g + 0.097906 * rgb.b;
    let u = 128.0 / 255.0 - 0.148223 * rgb.r - 0.290993 * rgb.g + 0.439216 * rgb.b;
    let v = 128.0 / 255.0 + 0.439216 * rgb.r - 0.367788 * rgb.g - 0.071427 * rgb.b;
    return vec3<f32>(y, u, v);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(input_img);
    if (id.x >= size.x || id.y >= size.y) {
        return;
    }

    let coord = vec2<i32>(i32(id.x), i32(id.y));
    let rgb = textureLoad(input_img, coord).rgb;
    let yuv = rgb_to_yuv(rgb);
    textureStore(y_plane, coord, vec4<f32>(yuv.x, 0.0, 0.0, 1.0));

    if ((id.x & 1u) == 0u && (id.y & 1u) == 0u) {
        let x1 = min(id.x + 1u, size.x - 1u);
        let y1 = min(id.y + 1u, size.y - 1u);
        let rgb00 = rgb;
        let rgb10 = textureLoad(input_img, vec2<i32>(i32(x1), i32(id.y))).rgb;
        let rgb01 = textureLoad(input_img, vec2<i32>(i32(id.x), i32(y1))).rgb;
        let rgb11 = textureLoad(input_img, vec2<i32>(i32(x1), i32(y1))).rgb;
        let uv = (rgb_to_yuv(rgb00).yz
            + rgb_to_yuv(rgb10).yz
            + rgb_to_yuv(rgb01).yz
            + rgb_to_yuv(rgb11).yz) * 0.25;
        textureStore(uv_plane, vec2<i32>(i32(id.x / 2u), i32(id.y / 2u)), vec4<f32>(uv, 0.0, 1.0));
    }
}
