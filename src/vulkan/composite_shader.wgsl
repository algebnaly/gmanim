@group(0) @binding(0) var background: texture_storage_2d<rgba8unorm, read_write>;
@group(0) @binding(1) var foreground: texture_2d<f32>;
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
    let dim = textureDimensions(background);
    if (global_id.x >= dim.x || global_id.y >= dim.y) {
        return;
    }
    
    let fg_dim = textureDimensions(foreground).xy;
    let ssaa_factor = fg_dim.x / dim.x;
    
    var fg_color_sum = vec4<f32>(0.0);
    let base_x = global_id.x * ssaa_factor;
    let base_y = global_id.y * ssaa_factor;
    
    for (var dy = 0u; dy < ssaa_factor; dy = dy + 1u) {
        for (var dx = 0u; dx < ssaa_factor; dx = dx + 1u) {
            let px = vec2<i32>(i32(base_x + dx), i32(base_y + dy));
            fg_color_sum += textureLoad(foreground, px, 0);
        }
    }
    
    let fg_hdr = fg_color_sum / f32(ssaa_factor * ssaa_factor);
    let bloom_size = textureDimensions(bloom_image);
    let bloom_coord = min(global_id.xy * bloom_size / dim, bloom_size - vec2<u32>(1u));
    let bloom = textureLoad(bloom_image, vec2<i32>(bloom_coord), 0).rgb;
    let fg_color = vec4<f32>(aces_tone_map(fg_hdr.rgb + bloom * 0.7), fg_hdr.a);
    let bg_color = textureLoad(background, vec2<i32>(global_id.xy));
    
    // Alpha blending
    let final_color = fg_color.rgb * fg_color.a + bg_color.rgb * (1.0 - fg_color.a);
    let final_alpha = fg_color.a + bg_color.a * (1.0 - fg_color.a);
    
    textureStore(background, vec2<i32>(global_id.xy), vec4<f32>(final_color, final_alpha));
}
