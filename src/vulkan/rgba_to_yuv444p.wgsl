@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> yuv_buf: array<u32>;

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
    let y = global_id.y; // 0 to height - 1

    let base_x = x * 4u;
    let base_y = y;

    if (base_x >= constants.width || base_y >= constants.height) {
        return;
    }

    // Read 4 pixels in a row
    let p0 = textureLoad(input_tex, vec2<i32>(i32(base_x), i32(base_y)), 0).rgb;
    let p1 = textureLoad(input_tex, vec2<i32>(i32(base_x + 1u), i32(base_y)), 0).rgb;
    let p2 = textureLoad(input_tex, vec2<i32>(i32(base_x + 2u), i32(base_y)), 0).rgb;
    let p3 = textureLoad(input_tex, vec2<i32>(i32(base_x + 3u), i32(base_y)), 0).rgb;

    // Y values
    let y_u32 = pack4x8unorm(vec4<f32>(
        rgb_to_y(p0), rgb_to_y(p1), rgb_to_y(p2), rgb_to_y(p3)
    ));

    // U values
    let u_u32 = pack4x8unorm(vec4<f32>(
        rgb_to_u(p0), rgb_to_u(p1), rgb_to_u(p2), rgb_to_u(p3)
    ));

    // V values
    let v_u32 = pack4x8unorm(vec4<f32>(
        rgb_to_v(p0), rgb_to_v(p1), rgb_to_v(p2), rgb_to_v(p3)
    ));

    let width_u32 = constants.width / 4u;
    let idx = base_y * width_u32 + x;
    
    let plane_size_u32 = constants.width * constants.height / 4u;

    yuv_buf[idx] = y_u32;
    yuv_buf[plane_size_u32 + idx] = u_u32;
    yuv_buf[plane_size_u32 * 2u + idx] = v_u32;
}
