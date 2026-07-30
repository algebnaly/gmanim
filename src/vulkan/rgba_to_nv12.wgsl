@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> nv12_buf: array<u32>;

struct Constants {
    width: u32,
    height: u32,
}
@group(0) @binding(2) var<uniform> constants: Constants;

fn rgb_to_y(color: vec3<f32>) -> f32 {
    return bt709_to_yuv_limited(color).x;
}

fn rgb_to_u(color: vec3<f32>) -> f32 {
    return bt709_to_yuv_limited(color).y;
}

fn rgb_to_v(color: vec3<f32>) -> f32 {
    return bt709_to_yuv_limited(color).z;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x; // 0 to width / 4 - 1
    let y = global_id.y; // 0 to height / 2 - 1

    let base_x = x * 4u;
    let base_y = y * 2u;

    if (base_x >= constants.width || base_y >= constants.height) {
        return;
    }

    // Read 8 pixels for the 4x2 block
    let p00 = textureLoad(input_tex, vec2<i32>(i32(base_x), i32(base_y)), 0).rgb;
    let p10 = textureLoad(input_tex, vec2<i32>(i32(base_x + 1u), i32(base_y)), 0).rgb;
    let p20 = textureLoad(input_tex, vec2<i32>(i32(base_x + 2u), i32(base_y)), 0).rgb;
    let p30 = textureLoad(input_tex, vec2<i32>(i32(base_x + 3u), i32(base_y)), 0).rgb;

    let p01 = textureLoad(input_tex, vec2<i32>(i32(base_x), i32(base_y + 1u)), 0).rgb;
    let p11 = textureLoad(input_tex, vec2<i32>(i32(base_x + 1u), i32(base_y + 1u)), 0).rgb;
    let p21 = textureLoad(input_tex, vec2<i32>(i32(base_x + 2u), i32(base_y + 1u)), 0).rgb;
    let p31 = textureLoad(input_tex, vec2<i32>(i32(base_x + 3u), i32(base_y + 1u)), 0).rgb;

    // Y values
    let y0_u32 = pack4x8unorm(vec4<f32>(
        rgb_to_y(p00), rgb_to_y(p10), rgb_to_y(p20), rgb_to_y(p30)
    ));
    let y1_u32 = pack4x8unorm(vec4<f32>(
        rgb_to_y(p01), rgb_to_y(p11), rgb_to_y(p21), rgb_to_y(p31)
    ));

    let width_u32 = constants.width / 4u;
    let y_idx0 = base_y * width_u32 + x;
    let y_idx1 = (base_y + 1u) * width_u32 + x;

    nv12_buf[y_idx0] = y0_u32;
    nv12_buf[y_idx1] = y1_u32;

    // UV values
    // Average 2x2 blocks for chroma subsampling
    let avg0 = (p00 + p10 + p01 + p11) * 0.25;
    let avg1 = (p20 + p30 + p21 + p31) * 0.25;

    let u0 = rgb_to_u(avg0);
    let v0 = rgb_to_v(avg0);
    let u1 = rgb_to_u(avg1);
    let v1 = rgb_to_v(avg1);

    let uv_u32 = pack4x8unorm(vec4<f32>(u0, v0, u1, v1));

    let uv_offset = (constants.width * constants.height) / 4u;
    let uv_idx = uv_offset + y * width_u32 + x;

    nv12_buf[uv_idx] = uv_u32;
}
