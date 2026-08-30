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
    origin: vec4<f32>,
    u_axis: vec4<f32>,
    v_axis: vec4<f32>,
    major_color: vec4<f32>,
    minor_color: vec4<f32>,
    u_axis_color: vec4<f32>,
    v_axis_color: vec4<f32>,
    params: vec4<f32>,
    extent: vec4<f32>,
}

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(0) @binding(1) var<storage, read> grids: array<GridData3D>;
@group(0) @binding(2) var sdf_depth: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) @interpolate(linear) edge_distance: f32,
    @location(1) @interpolate(perspective) local_position: vec2<f32>,
    @location(2) @interpolate(perspective) world_position: vec3<f32>,
    @location(3) @interpolate(flat) color: vec4<f32>,
    @location(4) @interpolate(flat) line_width: f32,
    @location(5) @interpolate(flat) line_alpha: f32,
    @location(6) @interpolate(flat) fade_radius: f32,
    @location(7) @interpolate(flat) lod_center: vec2<f32>,
    @location(8) @interpolate(flat) lod_half_span: f32,
    @location(9) @interpolate(flat) plane_normal: vec3<f32>,
    @location(10) @interpolate(flat) grid_index: u32,
    @location(11) @interpolate(flat) orientation: u32,
    @location(12) @interpolate(flat) lod_alphas: vec3<f32>,
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

fn local_camera_coordinate(delta: vec3<f32>, axis: vec3<f32>) -> f32 {
    return dot(delta, axis) / max(dot(axis, axis), 1e-8);
}

fn projected_spacing_pixels(
    world_position: vec3<f32>,
    offset: vec3<f32>,
    view: mat4x4<f32>,
) -> f32 {
    let a = project(world_position, view);
    let b = project(world_position + offset, view);
    if a.w <= 1e-4 || b.w <= 1e-4 {
        return 0.0;
    }
    let pixel_scale = vec2<f32>(camera.width, camera.height) * 0.5;
    return length((b.xy / b.w - a.xy / a.w) * pixel_scale);
}

fn disabled_vertex() -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(2.0, 2.0, 1.0, 1.0);
    out.edge_distance = 0.0;
    out.local_position = vec2<f32>(0.0);
    out.world_position = vec3<f32>(0.0);
    out.color = vec4<f32>(0.0);
    out.line_width = 0.0;
    out.line_alpha = 0.0;
    out.fade_radius = 0.0;
    out.lod_center = vec2<f32>(0.0);
    out.lod_half_span = 1.0;
    out.plane_normal = vec3<f32>(0.0, 0.0, 1.0);
    out.grid_index = 0u;
    out.orientation = 0u;
    out.lod_alphas = vec3<f32>(0.0);
    return out;
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    let grid_index = instance_index / INSTANCES_PER_GRID;
    let grid_instance = instance_index % INSTANCES_PER_GRID;
    let lines_per_orientation = LOD_COUNT * LINE_COUNT;
    let orientation = grid_instance / lines_per_orientation;
    let orientation_instance = grid_instance % lines_per_orientation;
    let lod = orientation_instance / LINE_COUNT;
    let line_index = orientation_instance % LINE_COUNT;
    let grid = grids[grid_index];

    let subdivisions = max(grid.params.y, 1.0);
    let minor_spacing = grid.params.x / subdivisions;
    let spacing = minor_spacing * pow(subdivisions, f32(lod));
    let next_spacing = spacing * subdivisions;
    let extent = grid.extent.x;
    let plane_normal = normalize(cross(grid.u_axis.xyz, grid.v_axis.xyz));
    let view_ray = normalize(camera.look_at);
    let plane_denominator = dot(plane_normal, view_ray);
    var focus_position = camera.pos;
    if abs(plane_denominator) > 1e-4 {
        let focus_distance = dot(grid.origin.xyz - camera.pos, plane_normal)
            / plane_denominator;
        if focus_distance > 0.0 {
            focus_position = camera.pos + view_ray * focus_distance;
        }
    }
    let camera_delta = focus_position - grid.origin.xyz;
    let camera_u = local_camera_coordinate(camera_delta, grid.u_axis.xyz);
    let camera_v = local_camera_coordinate(camera_delta, grid.v_axis.xyz);
    let line_center = select(camera_v, camera_u, orientation == 1u);
    let along_center = select(camera_u, camera_v, orientation == 1u);
    let line_coordinate = round(line_center / spacing) * spacing
        + (f32(line_index) - (f32(LINE_COUNT) - 1.0) * 0.5) * spacing;
    let half_span = min(extent, (f32(LINE_COUNT) - 1.0) * 0.5 * spacing);
    let along_start = max(-extent, along_center - half_span);
    let along_end = min(extent, along_center + half_span);

    if abs(line_coordinate) > extent || along_start >= along_end {
        return disabled_vertex();
    }
    if lod + 1u < LOD_COUNT {
        let coarse_coordinate = round(line_coordinate / next_spacing) * next_spacing;
        if abs(line_coordinate - coarse_coordinate) < spacing * 0.01 {
            return disabled_vertex();
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

    let focus_world = grid.origin.xyz
        + grid.u_axis.xyz * camera_u
        + grid.v_axis.xyz * camera_v;
    let minor_u_pixel_spacing = projected_spacing_pixels(
        focus_world,
        grid.u_axis.xyz * minor_spacing,
        view,
    );
    let minor_v_pixel_spacing = projected_spacing_pixels(
        focus_world,
        grid.v_axis.xyz * minor_spacing,
        view,
    );
    let minor_pixel_spacing = min(minor_u_pixel_spacing, minor_v_pixel_spacing);
    let lod_alphas = vec3<f32>(
        smoothstep(2.0, 8.0, minor_pixel_spacing),
        smoothstep(2.0, 8.0, minor_pixel_spacing * subdivisions),
        smoothstep(2.0, 8.0, minor_pixel_spacing * subdivisions * subdivisions),
    );
    let lod_alpha = lod_alphas[lod];

    let on_axis = abs(line_coordinate) < spacing * 0.01;
    var color = select(grid.major_color, grid.minor_color, lod == 0u);
    if on_axis {
        color = select(grid.u_axis_color, grid.v_axis_color, orientation == 1u);
    }

    let corner_index = array<u32, 6>(0u, 1u, 2u, 0u, 2u, 3u)[vertex_index];
    let at_end = corner_index == 1u || corner_index == 2u;
    let side = select(-1.0, 1.0, corner_index >= 2u);
    let clip = select(start_clip, end_clip, at_end);
    let world_position = select(start_world, end_world, at_end);
    let local_position = select(start_local, end_local, at_end);
    let start_ndc = start_clip.xy / start_clip.w;
    let end_ndc = end_clip.xy / end_clip.w;
    let screen_direction = (end_ndc - start_ndc) * vec2<f32>(camera.width, camera.height);
    if length(screen_direction) <= 1e-4 {
        return disabled_vertex();
    }
    let screen_normal = normalize(vec2<f32>(-screen_direction.y, screen_direction.x));
    let half_quad_width = max(grid.params.z * 0.5, 0.01) + 1.0;
    let ndc_offset = screen_normal
        * side
        * half_quad_width
        * 2.0
        / vec2<f32>(camera.width, camera.height);

    var out: VertexOutput;
    out.clip_position = vec4<f32>(clip.xy + ndc_offset * clip.w, clip.zw);
    // Interpolate distance in raster pixels so fragment coverage matches SSAA.
    out.edge_distance = side * half_quad_width * raster_pixel_scale();
    out.local_position = local_position;
    out.world_position = world_position;
    out.color = color;
    out.line_width = grid.params.z;
    out.line_alpha = lod_alpha;
    out.fade_radius = grid.params.w;
    out.lod_center = vec2<f32>(camera_u, camera_v);
    out.lod_half_span = half_span;
    out.plane_normal = plane_normal;
    out.grid_index = grid_index;
    out.orientation = orientation;
    out.lod_alphas = lod_alphas;
    return out;
}

fn raster_pixel_scale() -> f32 {
    return f32(max(camera.raster_scale, 1u));
}

fn line_coverage(pixel_distance: f32, line_width: f32) -> f32 {
    // `pixel_distance` is in raster pixels. SSAA shrinks those pixels, so
    // convert the output-pixel width the same way `raster_shader.wgsl` does.
    let scale = raster_pixel_scale();
    let target_w = max(line_width * scale, 0.01);
    let half_width = max(target_w * 0.5, 0.01);
    return 1.0 - smoothstep(
        max(half_width - 0.75 * scale, 0.0),
        half_width + 0.75 * scale,
        abs(pixel_distance),
    );
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if camera.num_primitives > 0u {
        let dimensions = textureDimensions(sdf_depth);
        let position = min(
            vec2<u32>(in.clip_position.xy) / max(camera.raster_scale, 1u),
            dimensions - vec2<u32>(1u),
        );
        let surface_depth = textureLoad(sdf_depth, vec2<i32>(position), 0).r;
        let grid_depth = dot(
            in.world_position - camera.pos,
            normalize(camera.look_at),
        );
        if surface_depth + 1e-4 < grid_depth {
            discard;
        }
    }

    let coverage = line_coverage(in.edge_distance, in.line_width);
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
    let fade_alpha = in.line_alpha
        * distance_alpha
        * lod_range_alpha
        * angle_alpha;
    var alpha = in.color.a * fade_alpha * coverage;

    // Every horizontal LOD is emitted before the vertical LODs. Reconstruct
    // their accumulated alpha and make the vertical source contribute only the
    // remainder required by the coverage union.
    let other_coordinate_gradient = length(vec2<f32>(
        dpdx(in.local_position.y),
        dpdy(in.local_position.y),
    ));
    if in.orientation == 1u && other_coordinate_gradient > 1e-8 {
        let grid = grids[in.grid_index];
        let subdivisions = max(grid.params.y, 1.0);
        let minor_spacing = grid.params.x / subdivisions;
        var other_alpha = 0.0;
        var other_spacing = minor_spacing;
        for (var other_lod = 0u; other_lod < LOD_COUNT; other_lod += 1u) {
            let other_coordinate = round(in.local_position.y / other_spacing) * other_spacing;
            let other_center = round(in.lod_center.y / other_spacing) * other_spacing;
            let other_half_span = min(
                grid.extent.x,
                (f32(LINE_COUNT) - 1.0) * 0.5 * other_spacing,
            );
            var other_line_exists = abs(other_coordinate - other_center)
                    <= other_half_span + other_spacing * 0.01
                && abs(in.local_position.x - in.lod_center.x) <= other_half_span
                && abs(other_coordinate) <= grid.extent.x;
            if other_lod + 1u < LOD_COUNT {
                let next_spacing = other_spacing * subdivisions;
                let coarse_coordinate = round(other_coordinate / next_spacing) * next_spacing;
                other_line_exists = other_line_exists
                    && abs(other_coordinate - coarse_coordinate) >= other_spacing * 0.01;
            }

            if other_line_exists {
                let local_distance = abs(in.local_position.y - other_coordinate);
                let other_pixel_distance = local_distance / other_coordinate_gradient;
                let other_coverage = line_coverage(other_pixel_distance, in.line_width);
                let other_lod_distance = lod_radius / max(other_half_span, 1e-4);
                let other_range_alpha = 1.0
                    - smoothstep(0.72, 1.0, other_lod_distance);
                var other_color = select(
                    grid.major_color,
                    grid.minor_color,
                    other_lod == 0u,
                );
                if abs(other_coordinate) < other_spacing * 0.01 {
                    other_color = grid.u_axis_color;
                }
                let candidate_alpha = other_color.a
                    * in.lod_alphas[other_lod]
                    * distance_alpha
                    * other_range_alpha
                    * angle_alpha
                    * other_coverage;
                other_alpha = candidate_alpha + other_alpha * (1.0 - candidate_alpha);
            }
            other_spacing *= subdivisions;
        }

        let union_alpha = max(other_alpha, alpha);
        alpha = select(
            0.0,
            (union_alpha - other_alpha) / max(1.0 - other_alpha, 1e-6),
            union_alpha > other_alpha + 1e-6,
        );
    }
    if alpha <= 1e-4 {
        discard;
    }
    return vec4<f32>(in.color.rgb, alpha);
}
