@group(0) @binding(0) var output_image: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var input_image: texture_2d<f32>;
@group(0) @binding(2) var bloom_image: texture_2d<f32>;

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

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_size = textureDimensions(output_image);
    if (global_id.x >= output_size.x || global_id.y >= output_size.y) {
        return;
    }

    let input_size = textureDimensions(input_image).xy;
    let factor = input_size.x / output_size.x;
    let base = global_id.xy * factor;
    var color = vec4<f32>(0.0);

    for (var y = 0u; y < factor; y = y + 1u) {
        for (var x = 0u; x < factor; x = x + 1u) {
            color += textureLoad(input_image, vec2<i32>(base + vec2<u32>(x, y)), 0);
        }
    }

    let averaged = color / f32(factor * factor);
    let bloom_size = textureDimensions(bloom_image);
    let bloom_coord = min(global_id.xy * bloom_size / output_size, bloom_size - vec2<u32>(1u));
    let bloom = textureLoad(bloom_image, vec2<i32>(bloom_coord), 0).rgb;
    textureStore(
        output_image,
        vec2<i32>(global_id.xy),
        vec4<f32>(aces_tone_map(averaged.rgb + bloom * 0.7), averaged.a),
    );
}
