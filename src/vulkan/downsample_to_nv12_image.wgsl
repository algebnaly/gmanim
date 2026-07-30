@group(0) @binding(0) var input_img: texture_2d<f32>;
@group(0) @binding(1) var y_plane: texture_storage_2d<r8unorm, write>;
@group(0) @binding(2) var uv_plane: texture_storage_2d<rg8unorm, write>;

fn average_2x2(base: vec2<u32>) -> vec3<f32> {
    let color = (
        textureLoad(input_img, vec2<i32>(base), 0).rgb
        + textureLoad(input_img, vec2<i32>(base + vec2<u32>(1u, 0u)), 0).rgb
        + textureLoad(input_img, vec2<i32>(base + vec2<u32>(0u, 1u)), 0).rgb
        + textureLoad(input_img, vec2<i32>(base + vec2<u32>(1u, 1u)), 0).rgb
    ) * 0.25;
    return hdr_to_bt709(color);
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
    let yuv00 = bt709_to_yuv_limited(rgb00);
    let yuv10 = bt709_to_yuv_limited(rgb10);
    let yuv01 = bt709_to_yuv_limited(rgb01);
    let yuv11 = bt709_to_yuv_limited(rgb11);

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
