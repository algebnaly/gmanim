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
    // The spare component of each geometry vector contains one per-frame invariant.
    origin: vec4<f32>,     // xyz: origin, w: half extent
    u_axis: vec4<f32>,     // xyz: basis, w: camera-local u
    v_axis: vec4<f32>,     // xyz: basis, w: camera-local v
    major_color: vec4<f32>,
    minor_color: vec4<f32>,
    u_axis_color: vec4<f32>,
    v_axis_color: vec4<f32>,
    params: vec4<f32>,     // line width, fade radius, lod alpha 0, lod alpha 1
    lod: vec4<f32>,        // lod spacings 0..2, lod alpha 2
}

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(0) @binding(1) var<storage, read> grids: array<GridData3D>;
@group(0) @binding(2) var sdf_depth: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) @interpolate(perspective) local_position: vec2<f32>,
    @location(1) @interpolate(perspective) world_position: vec3<f32>,
    @location(2) @interpolate(flat) color: vec4<f32>,
    @location(3) @interpolate(flat) line_alpha: f32,
    @location(4) @interpolate(flat) fade_radius: f32,
    @location(5) @interpolate(flat) lod_center: vec2<f32>,
    @location(6) @interpolate(flat) lod_half_span: f32,
    @location(7) @interpolate(flat) plane_normal: vec3<f32>,
    @location(8) @interpolate(flat) line_start: vec2<f32>,
    @location(9) @interpolate(flat) line_normal: vec2<f32>,
    @location(10) @interpolate(flat) line_width: f32,
}

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

fn screen_position(clip_position: vec4<f32>) -> vec2<f32> {
    let ndc = clip_position.xy / clip_position.w;
    let vulkan_ndc = vec2<f32>(ndc.x, -ndc.y);
    let raster_size = vec2<f32>(camera.width, camera.height)
        * f32(max(camera.raster_scale, 1u));
    return (vulkan_ndc * 0.5 + 0.5) * raster_size;
}

fn disabled_vertex() -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(2.0, 2.0, 1.0, 1.0);
    out.local_position = vec2<f32>(0.0);
    out.world_position = vec3<f32>(0.0);
    out.color = vec4<f32>(0.0);
    out.line_alpha = 0.0;
    out.fade_radius = 0.0;
    out.lod_center = vec2<f32>(0.0);
    out.lod_half_span = 1.0;
    out.plane_normal = vec3<f32>(0.0, 0.0, 1.0);
    out.line_start = vec2<f32>(0.0);
    out.line_normal = vec2<f32>(0.0, 1.0);
    out.line_width = 0.0;
    return out;
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    let grid_index = instance_index / INSTANCES_PER_GRID;
    let grid_instance = instance_index % INSTANCES_PER_GRID;
    let grid = grids[grid_index];
    let extent = grid.origin.w;
    let camera_u = grid.u_axis.w;
    let camera_v = grid.v_axis.w;
    let regular_count = INSTANCES_PER_GRID - AXIS_LINE_COUNT;

    var orientation: u32;
    var lod: u32 = 0u;
    var line_coordinate: f32;
    var along_start: f32;
    var along_end: f32;
    var half_span: f32;
    var is_world_axis: bool;

    if grid_instance >= regular_count {
        // Dedicated world-axis strokes. The sliding LOD window used to draw
        // these only on the coarsest level, so they vanished whenever that
        // level's alpha collapsed or the origin left the camera-centered set.
        orientation = grid_instance - regular_count;
        is_world_axis = true;
        line_coordinate = 0.0;
        along_start = -extent;
        along_end = extent;
        half_span = extent;
        if along_start >= along_end {
            return disabled_vertex();
        }
    } else {
        let lines_per_orientation = LOD_COUNT * LINE_COUNT;
        orientation = grid_instance / lines_per_orientation;
        let orientation_instance = grid_instance % lines_per_orientation;
        lod = orientation_instance / LINE_COUNT;
        let line_index = orientation_instance % LINE_COUNT;
        let spacing = grid.lod[lod];
        let next_spacing = grid.lod[min(lod + 1u, LOD_COUNT - 1u)];
        let line_center = select(camera_v, camera_u, orientation == 1u);
        let along_center = select(camera_u, camera_v, orientation == 1u);
        line_coordinate = round(line_center / spacing) * spacing
            + (f32(line_index) - (f32(LINE_COUNT) - 1.0) * 0.5) * spacing;
        half_span = min(extent, (f32(LINE_COUNT) - 1.0) * 0.5 * spacing);
        along_start = max(-extent, along_center - half_span);
        along_end = min(extent, along_center + half_span);
        is_world_axis = false;

        if abs(line_coordinate) > extent || along_start >= along_end {
            return disabled_vertex();
        }
        // Origin is owned by the dedicated axis instances.
        if abs(line_coordinate) < spacing * 0.01 {
            return disabled_vertex();
        }
        if lod + 1u < LOD_COUNT {
            let coarse_coordinate = round(line_coordinate / next_spacing) * next_spacing;
            if abs(line_coordinate - coarse_coordinate) < spacing * 0.01 {
                return disabled_vertex();
            }
        }
    }

    var start_local: vec2<f32>;
    var end_local: vec2<f32>;
    if orientation == 0u {
        start_local = vec2<f32>(along_start, line_coordinate);
        end_local = vec2<f32>(along_end, line_coordinate);
    } else {
        start_local = vec2<f32>(line_coordinate, along_start);
        end_local = vec2<f32>(line_coordinate, along_end);
    }

    var start_world = grid.origin.xyz
        + grid.u_axis.xyz * start_local.x
        + grid.v_axis.xyz * start_local.y;
    var end_world = grid.origin.xyz
        + grid.u_axis.xyz * end_local.x
        + grid.v_axis.xyz * end_local.y;
    let view = view_matrix();
    var start_clip = project(start_world, view);
    var end_clip = project(end_world, view);
    let near_w = 1e-3;
    if start_clip.w <= near_w && end_clip.w <= near_w {
        return disabled_vertex();
    }
    if start_clip.w <= near_w {
        let t = clamp((near_w - start_clip.w) / (end_clip.w - start_clip.w), 0.0, 1.0);
        start_world = mix(start_world, end_world, t);
        start_local = mix(start_local, end_local, t);
        start_clip = project(start_world, view);
    }
    if end_clip.w <= near_w {
        let t = clamp((near_w - end_clip.w) / (start_clip.w - end_clip.w), 0.0, 1.0);
        end_world = mix(end_world, start_world, t);
        end_local = mix(end_local, start_local, t);
        end_clip = project(end_world, view);
    }

    let at_end = vertex_index != 0u;
    let start_screen = screen_position(start_clip);
    let end_screen = screen_position(end_clip);
    let screen_direction = end_screen - start_screen;
    if length(screen_direction) <= 1e-4 {
        return disabled_vertex();
    }
    let line_normal = normalize(vec2<f32>(-screen_direction.y, screen_direction.x));
    let lod_alphas = vec3<f32>(grid.params.z, grid.params.w, grid.lod.w);
    var color = select(grid.major_color, grid.minor_color, lod == 0u);
    var line_alpha = lod_alphas[lod];
    if is_world_axis {
        color = select(grid.u_axis_color, grid.v_axis_color, orientation == 1u);
        line_alpha = 1.0;
    }

    var out: VertexOutput;
    out.clip_position = select(start_clip, end_clip, at_end);
    out.local_position = select(start_local, end_local, at_end);
    out.world_position = select(start_world, end_world, at_end);
    out.color = color;
    out.line_alpha = line_alpha;
    out.fade_radius = grid.params.y;
    if is_world_axis {
        out.lod_center = vec2<f32>(0.0);
        out.lod_half_span = max(extent, 1e-4);
    } else {
        out.lod_center = vec2<f32>(camera_u, camera_v);
        out.lod_half_span = half_span;
    }
    out.plane_normal = normalize(cross(grid.u_axis.xyz, grid.v_axis.xyz));
    out.line_start = start_screen;
    out.line_normal = line_normal;
    out.line_width = grid.params.x;
    return out;
}

fn line_coverage(pixel_distance: f32, line_width: f32) -> f32 {
    let scale = f32(max(camera.raster_scale, 1u));
    let half_width = max(line_width * 0.5, 0.01);
    return 1.0 - smoothstep(
        max(half_width - 0.75 * scale, 0.0),
        half_width + 0.75 * scale,
        abs(pixel_distance),
    );
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if camera.num_primitives > 0u {
        // sdf_depth is allocated at the raster resolution; map with a ratio
        // so the lookup stays correct if the resolutions ever diverge.
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
    if in.fade_radius > 0.0 {
        let radius = length(in.local_position) / in.fade_radius;
        distance_alpha = 1.0 - smoothstep(0.65, 1.0, radius);
    }
    let lod_radius = length(in.local_position - in.lod_center);
    let lod_distance = lod_radius / max(in.lod_half_span, 1e-4);
    let lod_range_alpha = 1.0 - smoothstep(0.72, 1.0, lod_distance);
    let view_direction = normalize(camera.pos - in.world_position);
    let angle_alpha = smoothstep(
        0.025,
        0.12,
        abs(dot(in.plane_normal, view_direction)),
    );
    let pixel_distance = dot(in.line_normal, in.clip_position.xy - in.line_start);
    let alpha = in.color.a
        * in.line_alpha
        * distance_alpha
        * lod_range_alpha
        * angle_alpha
        * line_coverage(pixel_distance, in.line_width);
    if alpha <= 1e-4 {
        discard;
    }
    // Premultiplied output: the grid pipeline accumulates with a MAX color
    // blend so overlapping line crossings keep single-line brightness
    // instead of double-blending into bright dots.
    return vec4<f32>(in.color.rgb * alpha, alpha);
}
