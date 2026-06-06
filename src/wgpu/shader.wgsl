@group(0) @binding(0) var output_tex: texture_storage_2d<rgba8unorm, write>;

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

struct PrimitiveData3D {
    color: vec4<f32>,
    params: array<f32, 12>,
    shape_type: u32,
    padding: array<u32, 3>,
}
@group(0) @binding(2) var<storage, read> primitives: array<PrimitiveData3D>;

struct MapResult {
    dist: f32,
    color: vec4<f32>,
}

fn sphere_3d(p: vec3<f32>, center: vec3<f32>, radius: f32) -> f32 {
    return length(p - center) - radius;
}

fn sd_box(p: vec3<f32>, b: vec3<f32>) -> f32 {
    let d = abs(p) - b;
    return length(max(d, vec3<f32>(0.0))) + min(max(d.x, max(d.y, d.z)), 0.0);
}

// Line segment/capsule SDF
fn line_3d(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, radius: f32) -> f32 {
    let pa = p - a;
    let ba = b - a;
    let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - radius;
}

fn sd_capped_cone(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, ra: f32, rb: f32) -> f32 {
    let rba = rb - ra;
    let ba = b - a;
    let pa = p - a;
    let baba = dot(ba, ba);
    let papa = dot(pa, pa);
    let paba = dot(pa, ba) / baba;
    
    let x = sqrt(max(0.0, papa - paba * paba * baba));
    let r_sel = select(rb, ra, paba < 0.5);
    let cax = max(0.0, x - r_sel);
    let cay = abs(paba - 0.5) - 0.5;
    
    let k = rba * rba + baba;
    let f = clamp((rba * (x - ra) + paba * baba) / k, 0.0, 1.0);
    
    let cbx = x - ra - f * rba;
    let cby = paba - f;
    
    let s = select(1.0, -1.0, cbx < 0.0 && cay < 0.0);
    
    return s * sqrt(min(cax * cax + cay * cay * baba, cbx * cbx + cby * cby * baba));
}

// Map function traversing all primitives
fn map(p: vec3<f32>) -> MapResult {
    var min_dist: f32 = 99999.0;
    var best_color: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    let num_primitives = arrayLength(&primitives);
    
    for (var i: u32 = 0u; i < num_primitives; i = i + 1u) {
        let prim = primitives[i];
        var d: f32 = 99999.0;
        
        if (prim.shape_type == 0u) { // Sphere
            // params: center.x, center.y, center.z, radius
            let center = vec3<f32>(prim.params[0], prim.params[1], prim.params[2]);
            let radius = prim.params[3];
            d = sphere_3d(p, center, radius);
        } else if (prim.shape_type == 1u) { // Line Segment
            // params: a.x, a.y, a.z, b.x, b.y, b.z, radius
            let a = vec3<f32>(prim.params[0], prim.params[1], prim.params[2]);
            let b = vec3<f32>(prim.params[3], prim.params[4], prim.params[5]);
            let radius = prim.params[6];
            d = line_3d(p, a, b, radius);
        } else if (prim.shape_type == 2u) { // Arrow
            let start = vec3<f32>(prim.params[0], prim.params[1], prim.params[2]);
            let end = vec3<f32>(prim.params[3], prim.params[4], prim.params[5]);
            let shaft_radius = prim.params[6];
            let head_radius = prim.params[7];
            let head_length = prim.params[8];
            
            let ba = end - start;
            let len = length(ba);
            if (len < 0.0001) {
                d = length(p - start) - shaft_radius;
            } else {
                let dir = ba / len;
                let head_base = end - dir * head_length;
                
                // Shaft
                let pa_s = p - start;
                let ba_s = head_base - start;
                let h_s = clamp(dot(pa_s, ba_s) / dot(ba_s, ba_s), 0.0, 1.0);
                let d_shaft = length(pa_s - ba_s * h_s) - shaft_radius;
                
                // Cone head
                let d_cone = sd_capped_cone(p, head_base, end, head_radius, 0.0);

                d = min(d_shaft, d_cone);
            }
        } else if (prim.shape_type == 3u) { // Box
            let center = vec3<f32>(prim.params[0], prim.params[1], prim.params[2]);
            let size = vec3<f32>(prim.params[3], prim.params[4], prim.params[5]);
            let x_axis = vec3<f32>(prim.params[6], prim.params[7], prim.params[8]);
            let y_axis = vec3<f32>(prim.params[9], prim.params[10], prim.params[11]);
            let z_axis = cross(x_axis, y_axis);
            
            let pt = p - center;
            let local_p = vec3<f32>(dot(pt, x_axis), dot(pt, y_axis), dot(pt, z_axis));
            
            d = sd_box(local_p, size);
        }
        
        if (d < min_dist) {
            min_dist = d;
            best_color = prim.color;
        }
    }
    
    return MapResult(min_dist, best_color);
}

fn calc_normal(p: vec3<f32>) -> vec3<f32> {
    let e = vec2<f32>(0.001, 0.0);
    let n = vec3<f32>(
        map(p + e.xyy).dist - map(p - e.xyy).dist,
        map(p + e.yxy).dist - map(p - e.yxy).dist,
        map(p + e.yyx).dist - map(p - e.yyx).dist
    );
    return normalize(n);
}

fn render_ray(ro: vec3<f32>, rd: vec3<f32>) -> vec4<f32> {
    var t: f32 = 0.0;
    let max_dist: f32 = 100.0;
    let surf_dist: f32 = 0.001;
    let max_steps: i32 = 100;
    
    var hit_color: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var hit = false;
    
    for (var i: i32 = 0; i < max_steps; i = i + 1) {
        let p = ro + rd * t;
        let res = map(p);
        if (res.dist < surf_dist) {
            hit = true;
            hit_color = res.color;
            break;
        }
        t = t + res.dist;
        if (t > max_dist) {
            break;
        }
    }
    
    var final_color = vec4<f32>(0.0, 0.0, 0.0, 0.0); // Transparent background by default
    
    if (hit) {
        let p = ro + rd * t;
        let normal = calc_normal(p);
        let light_pos = vec3<f32>(10.0, 10.0, 10.0); // Simple hardcoded light for now
        let light_dir = normalize(light_pos - p);
        
        let ambient = 0.2;
        let diff = max(dot(normal, light_dir), 0.0);
        let light_intensity = ambient + diff * 0.8;
        
        final_color = vec4<f32>(hit_color.rgb * light_intensity, hit_color.a);
    }
    
    return final_color;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dim = textureDimensions(output_tex);
    let x = global_id.x;
    let y = global_id.y;
    if (x >= dim.x || y >= dim.y) {
        return;
    }
    
    if (camera.has_clip == 1u) {
        if (f32(x) < camera.clip_x || f32(x) >= camera.clip_x + camera.clip_w ||
            f32(y) < camera.clip_y || f32(y) >= camera.clip_y + camera.clip_h) {
            textureStore(output_tex, vec2<i32>(i32(x), i32(y)), vec4<f32>(0.0, 0.0, 0.0, 0.0));
            return;
        }
    }

    let aspect = camera.width / camera.height;
    let fov_scale = tan(camera.fov * 0.5);
    
    // Camera coordinate system
    let cz = normalize(camera.look_at - camera.pos);
    let cx = normalize(cross(cz, camera.up));
    let cy = cross(cx, cz);
    let ro = camera.pos;

    let ndc_x = ((f32(x) + 0.5) / camera.width) * 2.0 - 1.0;
    let ndc_y = 1.0 - ((f32(y) + 0.5) / camera.height) * 2.0;
    
    var ro_sample = ro;
    var rd_sample = cz;
    
    if (camera.proj_type == 0u) {
        // Perspective
        rd_sample = normalize(cx * ndc_x * aspect * fov_scale + cy * ndc_y * fov_scale + cz);
    } else {
        // Orthographic
        let u = ndc_x * (camera.ortho_right - camera.ortho_left) * 0.5 + (camera.ortho_right + camera.ortho_left) * 0.5;
        let v = ndc_y * (camera.ortho_top - camera.ortho_bottom) * 0.5 + (camera.ortho_top + camera.ortho_bottom) * 0.5;
        ro_sample = ro + cx * u + cy * v;
        rd_sample = cz;
    }
    
    let sample_color = render_ray(ro_sample, rd_sample);
    let final_color = vec4<f32>(sample_color.rgb * sample_color.a, sample_color.a);
    
    // Un-premultiply alpha before storing (because output format is Rgba8Unorm and blending typically expects straight alpha, or if we want straight alpha out)
    // Wait, TinySkia uses premultiplied alpha (PremultipliedColorU8)!
    // If tiny_skia uses premultiplied alpha, then outputting premultiplied colors is correct!
    
    textureStore(output_tex, vec2<i32>(i32(x), i32(y)), final_color);
}
