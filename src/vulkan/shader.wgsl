@group(0) @binding(0) var normal_coverage_tex: texture_storage_2d<rgba16float, write>;

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
    raster_scale: u32,
    has_raster_surfaces: u32,
    proj_mat: mat4x4<f32>,
    light_pos: vec3<f32>,
    light_intensity: f32,
    light_color: vec3<f32>,
    environment_intensity: f32,
    environment_color: vec3<f32>,
    environment_rotation: f32,
    background_color: vec4<f32>,
}
@group(0) @binding(1) var<uniform> camera: CameraUniform;

struct PrimitiveData3D {
    material_index: u32,
    shape_type: u32,
    padding: array<u32, 2>,
    params: array<f32, 12>,
}
@group(0) @binding(2) var<storage, read> primitives: array<PrimitiveData3D>;

@group(0) @binding(3) var material_id_tex: texture_storage_2d<r32uint, write>;
@group(0) @binding(4) var depth_tex: texture_storage_2d<r32float, write>;

struct MapResult {
    dist: f32,
    material_index: u32,
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

fn sd_bezier(p: vec3<f32>, A: vec3<f32>, B: vec3<f32>, C: vec3<f32>, radius: f32) -> f32 {
    let k2 = A - 2.0 * B + C;
    let k1 = 2.0 * (B - A);
    let k0 = A - p;
    
    let a = dot(k2, k2);
    // If the curve is basically a straight line (a approx 0)
    if (a < 1e-6) {
        let pa = p - A;
        let ba = C - A;
        let h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
        return length(pa - ba * h) - radius;
    }
    
    let b = 3.0 * dot(k1, k2) / (2.0 * a);
    let c = (dot(k1, k1) + 2.0 * dot(k0, k2)) / (2.0 * a);
    let d = dot(k0, k1) / (2.0 * a);
    
    let p_cubic = c - b * b / 3.0;
    let q_cubic = d - b * c / 3.0 + 2.0 * b * b * b / 27.0;
    
    let p3 = p_cubic * p_cubic * p_cubic;
    let D = q_cubic * q_cubic / 4.0 + p3 / 27.0;
    
    var t_min: f32 = 0.0;
    
    if (D >= 0.0) {
        let sqrt_D = sqrt(D);
        let u = q_cubic / -2.0 + sqrt_D;
        let v = q_cubic / -2.0 - sqrt_D;
        
        let root_u = sign(u) * pow(abs(u), 1.0/3.0);
        let root_v = sign(v) * pow(abs(v), 1.0/3.0);
        
        var t = root_u + root_v - b / 3.0;
        t_min = clamp(t, 0.0, 1.0);
        
        // Check endpoints
        let pt0 = k0; // t=0
        let pt1 = k2 + k1 + k0; // t=1
        let pt_min = k2 * t_min * t_min + k1 * t_min + k0;
        
        let dist0 = dot(pt0, pt0);
        let dist1 = dot(pt1, pt1);
        let dist_min = dot(pt_min, pt_min);
        
        var min_sq_dist = dist_min;
        if (dist0 < min_sq_dist) { min_sq_dist = dist0; }
        if (dist1 < min_sq_dist) { min_sq_dist = dist1; }
        
        return sqrt(min_sq_dist) - radius;
    }
    
    let u = 2.0 * sqrt(-p_cubic / 3.0);
    let theta = acos(clamp(q_cubic / (2.0 * sqrt(-(p3 / 27.0))), -1.0, 1.0)) / 3.0;
    
    let t1 = u * cos(theta) - b / 3.0;
    let t2 = u * cos(theta + 2.09439510239) - b / 3.0;
    let t3 = u * cos(theta + 4.18879020479) - b / 3.0;
    
    let ct1 = clamp(t1, 0.0, 1.0);
    let ct2 = clamp(t2, 0.0, 1.0);
    let ct3 = clamp(t3, 0.0, 1.0);
    
    let p1 = k2 * ct1 * ct1 + k1 * ct1 + k0;
    let p2 = k2 * ct2 * ct2 + k1 * ct2 + k0;
    let p3_pt = k2 * ct3 * ct3 + k1 * ct3 + k0;
    
    let d1 = dot(p1, p1);
    let d2 = dot(p2, p2);
    let d3 = dot(p3_pt, p3_pt);
    
    if (d1 < d2 && d1 < d3) {
        t_min = ct1;
    } else if (d2 < d1 && d2 < d3) {
        t_min = ct2;
    } else {
        t_min = ct3;
    }
    
    let pt = k2 * t_min * t_min + k1 * t_min + k0;
    return length(pt) - radius;
}

// Map function traversing all primitives
fn map(p: vec3<f32>) -> MapResult {
    var min_dist: f32 = 99999.0;
    var material_index = 0u;
    let num_primitives = camera.num_primitives;
    
    for (var i = 0u; i < num_primitives; i = i + 1u) {
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
            // params: center.x, center.y, center.z, size.x, size.y, size.z,
            // x_axis.x, x_axis.y, x_axis.z, y_axis.x, y_axis.y, y_axis.z
            let center = vec3<f32>(prim.params[0], prim.params[1], prim.params[2]);
            let size = vec3<f32>(prim.params[3], prim.params[4], prim.params[5]);
            let x_axis = vec3<f32>(prim.params[6], prim.params[7], prim.params[8]);
            let y_axis = vec3<f32>(prim.params[9], prim.params[10], prim.params[11]);
            let z_axis = normalize(cross(x_axis, y_axis));
            let d_p = p - center;
            let local_p = vec3<f32>(dot(d_p, x_axis), dot(d_p, y_axis), dot(d_p, z_axis));
            d = sd_box(local_p, size);
        } else if (prim.shape_type == 4u) { // Quadratic Bezier
            let a = vec3<f32>(prim.params[0], prim.params[1], prim.params[2]);
            let b = vec3<f32>(prim.params[3], prim.params[4], prim.params[5]);
            let c = vec3<f32>(prim.params[6], prim.params[7], prim.params[8]);
            let radius = prim.params[9];
            d = sd_bezier(p, a, b, c, radius);
        }
        
        if (d < min_dist) {
            min_dist = d;
            material_index = prim.material_index;
        }
    }
    
    return MapResult(min_dist, material_index);
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

fn raster_height() -> f32 {
    return camera.height * f32(max(camera.raster_scale, 1u));
}

fn get_pixel_radius(t: f32) -> f32 {
    // The pixel footprint is measured in raster pixels so the analytic edge
    // feather spans one raster pixel regardless of the SSAA factor.
    if (camera.proj_type == 0u) {
        let fov_scale = tan(camera.fov * 0.5);
        return max(0.0001, (2.0 * t * fov_scale) / max(raster_height(), 1.0));
    } else {
        return max(0.0001, (camera.ortho_top - camera.ortho_bottom) / max(raster_height() * 2.0, 1.0));
    }
}

struct RayResult {
    normal: vec3<f32>,
    material_index: u32,
    linear_depth: f32,
    coverage: f32,
}

fn render_ray(ro: vec3<f32>, rd: vec3<f32>) -> RayResult {
    var t: f32 = 0.0;
    let max_dist: f32 = 100.0;
    let surf_dist: f32 = 0.001;
    let max_steps: i32 = 128;
    
    var min_norm_dist: f32 = 1e20;
    var best_t: f32 = 0.0;
    var best_material_index: u32 = 0u;
    var hit = false;
    
    for (var i: i32 = 0; i < max_steps; i = i + 1) {
        let p = ro + rd * t;
        let res = map(p);
        let d = res.dist;
        let pr = get_pixel_radius(t);
        
        if (d < surf_dist) {
            hit = true;
            best_t = t;
            best_material_index = res.material_index;
            break;
        }
        
        let norm_d = d / pr;
        if (norm_d < min_norm_dist) {
            min_norm_dist = norm_d;
            best_t = t;
            best_material_index = res.material_index;
        }
        
        t = t + max(d, surf_dist);
        if (t > max_dist) {
            break;
        }
    }
    
    var coverage: f32 = 0.0;
    if (hit) {
        coverage = 1.0;
    } else if (min_norm_dist < 1.0) {
        // Analytic distance feathering across the 1-pixel boundary
        coverage = clamp(1.0 - min_norm_dist, 0.0, 1.0);
    }
    
    if (coverage <= 0.0) {
        return RayResult(vec3<f32>(0.0), 0u, 1e20, 0.0);
    }

    let p = ro + rd * best_t;
    let normal = calc_normal(p);
    let linear_depth = dot(p - camera.pos, normalize(camera.look_at));
    return RayResult(normal, best_material_index, linear_depth, coverage);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dim = textureDimensions(normal_coverage_tex);
    let x = global_id.x;
    let y = global_id.y;
    if (x >= dim.x || y >= dim.y) {
        return;
    }
    
    let scale = f32(max(camera.raster_scale, 1u));
    let raster_width = camera.width * scale;
    let raster_h = raster_height();

    if (camera.has_clip == 1u) {
        // Clip rectangles are expressed in output pixels; the dispatch runs
        // at the SSAA raster resolution.
        let clip_x = camera.clip_x * scale;
        let clip_y = camera.clip_y * scale;
        let clip_w = camera.clip_w * scale;
        let clip_h = camera.clip_h * scale;
        if (f32(x) < clip_x || f32(x) >= clip_x + clip_w ||
            f32(y) < clip_y || f32(y) >= clip_y + clip_h) {
            textureStore(normal_coverage_tex, vec2<i32>(i32(x), i32(y)), vec4<f32>(0.0));
            textureStore(material_id_tex, vec2<i32>(i32(x), i32(y)), vec4<u32>(0u));
            textureStore(depth_tex, vec2<i32>(i32(x), i32(y)), vec4<f32>(1e20));
            return;
        }
    }

    let aspect = camera.width / camera.height;
    let fov_scale = tan(camera.fov * 0.5);
    
    // Camera coordinate system
    let cz = camera.look_at;
    let cx = normalize(cross(cz, camera.up));
    let cy = cross(cx, cz);
    let ro = camera.pos;

    let aa = max(camera.aa_level, 1u);
    var total_coverage: f32 = 0.0;
    var nearest_depth = 1e20;
    var nearest_material = 0u;
    var nearest_normal = vec3<f32>(0.0);

    for (var i = 0u; i < aa; i = i + 1u) {
        for (var j = 0u; j < aa; j = j + 1u) {
            // subpixel offset from -0.5 to 0.5
            let sub_x = (f32(i) + 0.5) / f32(aa) - 0.5;
            let sub_y = (f32(j) + 0.5) / f32(aa) - 0.5;

            let ndc_x = ((f32(x) + 0.5 + sub_x) / raster_width) * 2.0 - 1.0;
            let ndc_y = 1.0 - ((f32(y) + 0.5 + sub_y) / raster_h) * 2.0;
            
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
            
            let sample_result = render_ray(ro_sample, rd_sample);
            if (sample_result.coverage > 0.0) {
                total_coverage += sample_result.coverage;
                if (sample_result.linear_depth < nearest_depth) {
                    nearest_depth = sample_result.linear_depth;
                    nearest_material = sample_result.material_index;
                    nearest_normal = sample_result.normal;
                }
            }
        }
    }

    let sample_count = aa * aa;
    let coverage = clamp(total_coverage / f32(sample_count), 0.0, 1.0);
    textureStore(
        normal_coverage_tex,
        vec2<i32>(i32(x), i32(y)),
        vec4<f32>(nearest_normal, coverage),
    );
    textureStore(
        material_id_tex,
        vec2<i32>(i32(x), i32(y)),
        vec4<u32>(nearest_material, 0u, 0u, 0u),
    );
    textureStore(depth_tex, vec2<i32>(i32(x), i32(y)), vec4<f32>(nearest_depth));
}
