@group(0) @binding(0) var background: texture_storage_2d<rgba8unorm, read_write>;
@group(0) @binding(1) var foreground: texture_2d<f32>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dim = textureDimensions(background);
    if (global_id.x >= dim.x || global_id.y >= dim.y) {
        return;
    }
    
    let fg_color = textureLoad(foreground, vec2<i32>(global_id.xy), 0);
    let bg_color = textureLoad(background, vec2<i32>(global_id.xy));
    
    // Alpha blending
    let final_color = fg_color.rgb * fg_color.a + bg_color.rgb * (1.0 - fg_color.a);
    let final_alpha = fg_color.a + bg_color.a * (1.0 - fg_color.a);
    
    textureStore(background, vec2<i32>(global_id.xy), vec4<f32>(final_color, final_alpha));
}
