@group(0) @binding(0) var output_hdr: texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var sdf_hdr: texture_2d<f32>;
@group(0) @binding(2) var raster_hdr: texture_2d<f32>;
@group(0) @binding(3) var overlay_hdr: texture_2d<f32>;

fn scaled_load(texture: texture_2d<f32>, position: vec2<u32>, output_size: vec2<u32>) -> vec4<f32> {
    let input_size = textureDimensions(texture);
    let input_position = min(position * input_size / output_size, input_size - vec2<u32>(1));
    return textureLoad(texture, vec2<i32>(input_position), 0);
}

@compute @workgroup_size(16, 16, 1)
fn resolve_sdf(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_size = textureDimensions(output_hdr);
    if any(global_id.xy >= output_size) {
        return;
    }
    textureStore(
        output_hdr,
        vec2<i32>(global_id.xy),
        scaled_load(sdf_hdr, global_id.xy, output_size),
    );
}

@compute @workgroup_size(16, 16, 1)
fn resolve_raster(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_size = textureDimensions(output_hdr);
    if any(global_id.xy >= output_size) {
        return;
    }
    textureStore(
        output_hdr,
        vec2<i32>(global_id.xy),
        scaled_load(raster_hdr, global_id.xy, output_size),
    );
}

fn over(foreground: vec4<f32>, background: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        foreground.rgb + background.rgb * (1.0 - foreground.a),
        foreground.a + background.a * (1.0 - foreground.a),
    );
}

@compute @workgroup_size(16, 16, 1)
fn resolve_raster_overlay(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_size = textureDimensions(output_hdr);
    if any(global_id.xy >= output_size) {
        return;
    }
    let raster = scaled_load(raster_hdr, global_id.xy, output_size);
    let overlay = scaled_load(overlay_hdr, global_id.xy, output_size);
    textureStore(
        output_hdr,
        vec2<i32>(global_id.xy),
        over(overlay, raster),
    );
}

@compute @workgroup_size(16, 16, 1)
fn merge(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_size = textureDimensions(output_hdr);
    if any(global_id.xy >= output_size) {
        return;
    }
    let sdf = scaled_load(sdf_hdr, global_id.xy, output_size);
    let raster = scaled_load(raster_hdr, global_id.xy, output_size);
    textureStore(output_hdr, vec2<i32>(global_id.xy), over(raster, sdf));
}

@compute @workgroup_size(16, 16, 1)
fn merge_overlay(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_size = textureDimensions(output_hdr);
    if any(global_id.xy >= output_size) {
        return;
    }
    let sdf = scaled_load(sdf_hdr, global_id.xy, output_size);
    let raster = scaled_load(raster_hdr, global_id.xy, output_size);
    let overlay = scaled_load(overlay_hdr, global_id.xy, output_size);
    textureStore(
        output_hdr,
        vec2<i32>(global_id.xy),
        over(overlay, over(raster, sdf)),
    );
}
