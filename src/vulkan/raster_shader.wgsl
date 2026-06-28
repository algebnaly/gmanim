struct CameraUniform {
    pos: vec3<f32>,
    _padding0: u32,
    look_at: vec3<f32>,
    _padding1: u32,
    up: vec3<f32>,
    fov: f32,
    width: f32,
    height: f32,
    proj_type: u32,
    ortho_left: f32,
    ortho_right: f32,
    ortho_bottom: f32,
    ortho_top: f32,
    has_clip: u32,
    clip_x: f32,
    clip_y: f32,
    clip_w: f32,
    clip_h: f32,
    aa_level: u32,
    num_primitives: u32,
    _pad2: u32,
    _pad3: u32,
    proj_mat: mat4x4<f32>,
}
@group(0) @binding(1) var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) frag_pos: vec3<f32>,
}

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    
    // View matrix
    let w = -normalize(camera.look_at);
    let u = normalize(cross(camera.up, w));
    let v = cross(w, u);
    let view_mat = mat4x4<f32>(
        vec4<f32>(u.x, v.x, w.x, 0.0),
        vec4<f32>(u.y, v.y, w.y, 0.0),
        vec4<f32>(u.z, v.z, w.z, 0.0),
        vec4<f32>(-dot(u, camera.pos), -dot(v, camera.pos), -dot(w, camera.pos), 1.0)
    );
    
    let proj_mat = camera.proj_mat;
    
    let world_pos = vec4<f32>(model.position, 1.0);
    out.frag_pos = world_pos.xyz;
    let view_pos = view_mat * world_pos;
    out.clip_position = proj_mat * view_pos;
    // out.clip_position.z = out.clip_position.w * 0.5;
    out.normal = model.normal;
    out.color = model.color;
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput, @builtin(front_facing) is_front: bool) -> @location(0) vec4<f32> {
    let view_dir = -normalize(camera.look_at);
    var up = vec3<f32>(0.0, 1.0, 0.0);
    if (abs(view_dir.y) > 0.99) {
        up = vec3<f32>(1.0, 0.0, 0.0);
    }
    let right = normalize(cross(view_dir, up));
    let cam_up = cross(right, view_dir);

    // 3-point lighting setup relative to camera
    let key_light_dir = normalize(view_dir + right * 0.8 + cam_up * 0.6);
    let fill_light_dir = normalize(view_dir - right * 0.8 - cam_up * 0.2);
    
    let ambient = 0.35;
    
    let norm = normalize(in.normal);
    let final_norm = select(-norm, norm, is_front);
    
    let diff_key = max(dot(final_norm, key_light_dir), 0.0);
    let diff_fill = max(dot(final_norm, fill_light_dir), 0.0);
    
    let frag_view_dir = normalize(camera.pos - in.frag_pos);
    let reflect_dir = reflect(-key_light_dir, final_norm);
    let spec = pow(max(dot(frag_view_dir, reflect_dir), 0.0), 32.0) * 0.4;
    
    let result = (ambient + diff_key * 0.6 + diff_fill * 0.3 + spec) * in.color.rgb;
    return vec4<f32>(result, in.color.a);
}