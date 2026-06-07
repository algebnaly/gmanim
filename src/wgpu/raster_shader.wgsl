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
    _padding2: vec4<u32>,
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
    let w = normalize(camera.pos - camera.look_at);
    let u = normalize(cross(camera.up, w));
    let v = cross(w, u);
    let view_mat = mat4x4<f32>(
        vec4<f32>(u.x, v.x, w.x, 0.0),
        vec4<f32>(u.y, v.y, w.y, 0.0),
        vec4<f32>(u.z, v.z, w.z, 0.0),
        vec4<f32>(-dot(u, camera.pos), -dot(v, camera.pos), -dot(w, camera.pos), 1.0)
    );
    
    let fov = camera.fov;
    let aspect = camera.width / camera.height;
    let n = 0.1;
    let f = 1000.0;
    
    // Proj matrix (Perspective by default)
    // wgpu uses z in [0, 1] range.
    var proj_mat = mat4x4<f32>(
        vec4<f32>(1.0 / (aspect * tan(fov / 2.0)), 0.0, 0.0, 0.0),
        vec4<f32>(0.0, 1.0 / tan(fov / 2.0), 0.0, 0.0),
        vec4<f32>(0.0, 0.0, f / (n - f), -1.0),
        vec4<f32>(0.0, 0.0, (f * n) / (n - f), 0.0)
    );
    
    if camera.proj_type == 1u {
        // Ortho
        let l = camera.ortho_left;
        let r = camera.ortho_right;
        let b = camera.ortho_bottom;
        let t = camera.ortho_top;
        proj_mat = mat4x4<f32>(
            vec4<f32>(2.0 / (r - l), 0.0, 0.0, 0.0),
            vec4<f32>(0.0, 2.0 / (t - b), 0.0, 0.0),
            vec4<f32>(0.0, 0.0, 1.0 / (n - f), 0.0),
            vec4<f32>(-(r + l)/(r - l), -(t + b)/(t - b), n / (n - f), 1.0)
        );
    }
    
    let world_pos = vec4<f32>(model.position, 1.0);
    out.frag_pos = world_pos.xyz;
    out.clip_position = proj_mat * view_mat * world_pos;
    out.normal = model.normal;
    out.color = model.color;
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let light_pos = vec3<f32>(10.0, 10.0, 10.0);
    let light_color = vec3<f32>(1.0, 1.0, 1.0);
    let ambient = 0.2;
    
    let norm = normalize(in.normal);
    let light_dir = normalize(light_pos - in.frag_pos);
    let diff = max(dot(norm, light_dir), 0.0);
    
    let view_dir = normalize(camera.pos - in.frag_pos);
    let reflect_dir = reflect(-light_dir, norm);
    let spec = pow(max(dot(view_dir, reflect_dir), 0.0), 32.0);
    
    let result = (ambient + diff + spec) * in.color.rgb;
    return vec4<f32>(result, in.color.a);
}