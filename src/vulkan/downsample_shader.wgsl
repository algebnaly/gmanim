@group(0) @binding(0) var output_image: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var input_image: texture_2d<f32>;

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

    textureStore(
        output_image,
        vec2<i32>(global_id.xy),
        color / f32(factor * factor),
    );
}
