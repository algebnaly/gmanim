@group(0) @binding(0) var background: texture_storage_2d<rgba8unorm, read_write>;
@group(0) @binding(1) var foreground: texture_2d<f32>;

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
    
    let fg_color = fg_color_sum / f32(ssaa_factor * ssaa_factor);
    let bg_color = textureLoad(background, vec2<i32>(global_id.xy));
    
    // Alpha blending
    let final_color = fg_color.rgb * fg_color.a + bg_color.rgb * (1.0 - fg_color.a);
    let final_alpha = fg_color.a + bg_color.a * (1.0 - fg_color.a);
    
    textureStore(background, vec2<i32>(global_id.xy), vec4<f32>(final_color, final_alpha));
}
