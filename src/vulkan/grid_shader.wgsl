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

struct GridData3D {
    origin: vec4<f32>,     // xyz: world origin, w: half extent
    u_axis: vec4<f32>,     // xyz: world U basis
    v_axis: vec4<f32>,     // xyz: world V basis
    major_color: vec4<f32>,
    minor_color: vec4<f32>,
    u_axis_color: vec4<f32>,
    v_axis_color: vec4<f32>,
    params: vec4<f32>,     // line width px, fade radius, cell size, subdivisions
}

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(0) @binding(1) var<storage, read> grids: array<GridData3D>;
@group(0) @binding(2) var sdf_depth: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) @interpolate(perspective) local_position: vec2<f32>,
    @location(1) @interpolate(perspective) world_position: vec3<f32>,
    @location(2) @interpolate(flat) grid_index: u32,
    @location(3) @interpolate(flat) plane_normal: vec3<f32>,
}

const QUAD_CORNERS: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(1.0, -1.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, 1.0),
);

fn view_matrix() -> mat4x4<f32> {
    let w = -normalize(camera.look_at);
    let u = normalize(cross(camera.up, w));
    let v = cross(w, u);
    return mat4x4<f32>(
        vec4<f32>(u.x, v.x, w.x, 0.0),
        vec4<f32>(u.y, v.y, w.y, 0.0),
        vec4<f32>(u.z, v.z, w.z, 0.0),
        vec4<f32>(-dot(u, camera.pos), -dot(v, camera.pos), -dot(w, camera.pos), 1.0),
    );
}

fn project(world_position: vec3<f32>, view: mat4x4<f32>) -> vec4<f32> {
    return camera.proj_mat * view * vec4<f32>(world_position, 1.0);
}

fn disabled_vertex() -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(2.0, 2.0, 1.0, 1.0);
    out.local_position = vec2<f32>(0.0);
    out.world_position = vec3<f32>(0.0);
    out.grid_index = 0u;
    out.plane_normal = vec3<f32>(0.0, 0.0, 1.0);
    return out;
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    let grid = grids[instance_index];
    let extent = grid.origin.w;
    if extent <= 0.0 {
        return disabled_vertex();
    }

    let corner = QUAD_CORNERS[vertex_index];
    let local = corner * extent;
    let world = grid.origin.xyz
        + grid.u_axis.xyz * local.x
        + grid.v_axis.xyz * local.y;

    var out: VertexOutput;
    out.clip_position = project(world, view_matrix());
    out.local_position = local;
    out.world_position = world;
    out.grid_index = instance_index;
    out.plane_normal = normalize(cross(grid.u_axis.xyz, grid.v_axis.xyz));
    return out;
}

fn wrap_dist(coord: f32, spacing: f32) -> f32 {
    return abs(coord - spacing * round(coord / spacing));
}

fn line_coverage(distance: f32, pixel_size: f32, line_width: f32) -> f32 {
    let scale = f32(max(camera.raster_scale, 1u));
    let half_width = max(line_width * 0.5, 0.01);
    let pixel_distance = distance / max(pixel_size, 1e-8);
    return 1.0 - smoothstep(
        max(half_width - 0.75 * scale, 0.0),
        half_width + 0.75 * scale,
        pixel_distance,
    );
}

fn lod_alpha(spacing: f32, pixel_size: f32) -> f32 {
    return smoothstep(2.0, 8.0, spacing / max(pixel_size, 1e-8));
}

fn premult(color: vec4<f32>, coverage: f32) -> vec4<f32> {
    let alpha = color.a * coverage;
    return vec4<f32>(color.rgb * alpha, alpha);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let grid = grids[in.grid_index];
    if camera.num_primitives > 0u {
        let dimensions = textureDimensions(sdf_depth);
        let scale = f32(max(camera.raster_scale, 1u));
        let raster_size = vec2<u32>(
            max(u32(camera.width * scale), 1u),
            max(u32(camera.height * scale), 1u),
        );
        let position = min(
            vec2<u32>(in.clip_position.xy) * dimensions / raster_size,
            dimensions - vec2<u32>(1u),
        );
        let surface_depth = textureLoad(sdf_depth, vec2<i32>(position), 0).r;
        let grid_depth = dot(in.world_position - camera.pos, normalize(camera.look_at));
        if surface_depth + 1e-4 < grid_depth {
            discard;
        }
    }

    var distance_alpha = 1.0;
    if grid.params.y > 0.0 {
        let radius = length(in.local_position) / grid.params.y;
        distance_alpha = 1.0 - smoothstep(0.65, 1.0, radius);
    }
    let view_direction = normalize(camera.pos - in.world_position);
    let angle_alpha = smoothstep(
        0.025,
        0.12,
        abs(dot(in.plane_normal, view_direction)),
    );
    let fade = distance_alpha * angle_alpha;
    if fade <= 1e-4 {
        discard;
    }

    let uv = in.local_position;
    let du = fwidth(uv.x);
    let dv = fwidth(uv.y);
    // LOD is measured in output pixels; coverage still uses raster derivatives.
    let output_pixel_scale = f32(max(camera.raster_scale, 1u));
    let pixel_size = max(min(du, dv) * output_pixel_scale, 1e-8);
    let line_width = grid.params.x;
    let cell = max(grid.params.z, 1e-6);
    let subdivisions = max(grid.params.w, 1.0);
    let minor_spacing = cell / subdivisions;

    var color = vec4<f32>(0.0);
    if subdivisions > 1.5 {
        let minor = lod_alpha(minor_spacing, pixel_size) * max(
            line_coverage(wrap_dist(uv.x, minor_spacing), du, line_width),
            line_coverage(wrap_dist(uv.y, minor_spacing), dv, line_width),
        );
        color = max(color, premult(grid.minor_color, minor));
    }
    let major = lod_alpha(cell, pixel_size) * max(
        line_coverage(wrap_dist(uv.x, cell), du, line_width),
        line_coverage(wrap_dist(uv.y, cell), dv, line_width),
    );
    color = max(color, premult(grid.major_color, major));
    // World axes are the U=0 / V=0 lines of this plane, not camera-relative.
    color = max(color, premult(
        grid.v_axis_color,
        line_coverage(abs(uv.x), du, line_width),
    ));
    color = max(color, premult(
        grid.u_axis_color,
        line_coverage(abs(uv.y), dv, line_width),
    ));
    color *= fade;
    if color.a <= 1e-4 {
        discard;
    }
    return color;
}
