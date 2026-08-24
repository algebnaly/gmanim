use gmanim_core::{
    Color, RendererConfig, Scene, SceneConfig,
    mobjects::{
        GridPlane, GridPlane3D, GridStyle3D, Rectangle,
        mesh_3d::{SurfaceMaterial, TriangleMesh3D},
        object_3d::Sphere3D,
    },
    vulkan::{
        context::VulkanContext,
        renderer::{RenderOutputs, RendererStats, VulkanRenderer},
    },
};

fn render_and_count_non_black(
    renderer: &mut VulkanRenderer,
    scene: &Scene,
    config: &SceneConfig,
) -> (RendererStats, usize) {
    renderer.render_scene_with_outputs(scene, config, None, RenderOutputs::CPU_RGBA_ONLY);
    let stats = renderer.last_stats();
    let non_black = renderer
        .get_rgba_bytes()
        .unwrap()
        .chunks_exact(4)
        .filter(|pixel| pixel[0] > 8 || pixel[1] > 8 || pixel[2] > 8)
        .count();
    (stats, non_black)
}

fn render_coverage(renderer: &mut VulkanRenderer, scene: &Scene, config: &SceneConfig) -> u64 {
    renderer.render_scene_with_outputs(scene, config, None, RenderOutputs::CPU_RGBA_ONLY);
    renderer
        .get_rgba_bytes()
        .unwrap()
        .chunks_exact(4)
        .map(|pixel| u64::from(pixel[3]))
        .sum()
}

fn main() {
    let config = SceneConfig {
        width: 16.0,
        height: 9.0,
        output_width: 320,
        output_height: 180,
        scale_factor: 20.0,
        framerate: 60,
    };
    let context = VulkanContext::new().unwrap();
    let mut renderer = VulkanRenderer::new(
        context.clone(),
        RendererConfig {
            msaa_samples: 4,
            ssaa_factor: 1,
            output_color_profile: Default::default(),
            analytic_aa_2d: true,
        },
    );

    let empty = Scene::default();
    let (stats, non_black) = render_and_count_non_black(&mut renderer, &empty, &config);
    assert_eq!(stats.sdf_dispatches, 0);
    assert_eq!(stats.raster_passes, 0);
    assert_eq!(stats.surface_resolve_dispatches, 0);
    assert_eq!(stats.surface_lighting_dispatches, 0);
    assert_eq!(stats.surface_composite_dispatches, 0);
    assert_eq!(non_black, 0);

    let mut sdf = Scene::default();
    let sphere = Sphere3D {
        radius: 1.0,
        material: SurfaceMaterial {
            base_color: [0.86, 0.31, 0.24, 1.0],
            ..Default::default()
        },
    };
    let sphere = sdf.add_named("verify-sphere", sphere);
    sdf.world
        .get_mut(sphere)
        .unwrap()
        .move_by(nalgebra::Vector3::new(0.0, 0.0, -3.0));
    let (stats, non_black) = render_and_count_non_black(&mut renderer, &sdf, &config);
    assert_eq!(stats.sdf_dispatches, 1);
    assert_eq!(stats.raster_passes, 0);
    assert_eq!(stats.surface_resolve_dispatches, 1);
    assert_eq!(stats.surface_lighting_dispatches, 1);
    assert_eq!(stats.surface_composite_dispatches, 1);
    assert!(non_black > 100);

    let mut raster = Scene::default();
    let rectangle = Rectangle {
        p0: nalgebra::Point3::new(-2.0, -1.0, 0.0),
        p1: nalgebra::Point3::new(2.0, -1.0, 0.0),
        p2: nalgebra::Point3::new(2.0, 1.0, 0.0),
        p3: nalgebra::Point3::new(-2.0, 1.0, 0.0),
        color: Color::new(40, 180, 240, 255),
        ..Default::default()
    };
    let rectangle = raster.add_rectangle(rectangle);
    let (stats, non_black) = render_and_count_non_black(&mut renderer, &raster, &config);
    assert_eq!(stats.sdf_dispatches, 0);
    assert_eq!(stats.raster_passes, 1);
    assert_eq!(stats.downsample_dispatches, 0);
    assert_eq!(stats.surface_resolve_dispatches, 0);
    assert_eq!(stats.surface_lighting_dispatches, 0);
    assert_eq!(stats.surface_composite_dispatches, 0);
    assert!(non_black > 100);
    let original_non_black = non_black;

    raster
        .world
        .set_rectangle_corners(
            rectangle,
            [
                nalgebra::Point3::new(-3.0, -0.25, 0.0),
                nalgebra::Point3::new(3.0, -0.25, 0.0),
                nalgebra::Point3::new(1.0, 0.25, 0.0),
                nalgebra::Point3::new(-1.0, 0.25, 0.0),
            ],
        )
        .unwrap();
    let (stats, non_black) = render_and_count_non_black(&mut renderer, &raster, &config);
    assert_eq!(stats.mesh_2d_geometry_uploads, 1);
    assert_eq!(stats.mesh_2d_arena_rebuilds, 0);
    assert!(non_black < original_non_black);

    let mut mixed = sdf;
    let overlay = Rectangle {
        p0: nalgebra::Point3::new(-0.7, -0.2, 0.0),
        p1: nalgebra::Point3::new(0.7, -0.2, 0.0),
        p2: nalgebra::Point3::new(0.7, 0.2, 0.0),
        p3: nalgebra::Point3::new(-0.7, 0.2, 0.0),
        color: Color::new(30, 220, 120, 255),
        ..Default::default()
    };
    mixed.add_rectangle(overlay);
    let (stats, non_black) = render_and_count_non_black(&mut renderer, &mixed, &config);
    assert_eq!(stats.sdf_dispatches, 1);
    assert_eq!(stats.raster_passes, 1);
    assert_eq!(stats.surface_resolve_dispatches, 1);
    assert_eq!(stats.surface_lighting_dispatches, 1);
    assert_eq!(stats.surface_composite_dispatches, 1);
    assert!(non_black > 100);
    let sdf_sample =
        ((config.output_height / 2 + 12) * config.output_width + config.output_width / 2) as usize;
    let sdf_pixel = &renderer.get_rgba_bytes().unwrap()[sdf_sample * 4..sdf_sample * 4 + 4];
    assert!(
        sdf_pixel[0] > sdf_pixel[1],
        "2D overlay composition must preserve the SDF surface outside the overlay"
    );

    let depth_scene = |mesh_z| {
        let mut scene = Scene::default();
        let sphere = Sphere3D {
            radius: 1.0,
            material: SurfaceMaterial {
                base_color: [0.9, 0.08, 0.03, 1.0],
                emissive: [1.0, 0.0, 0.0],
                emissive_strength: 1.0,
                ..Default::default()
            },
        };
        let sphere = scene.add_named("depth-sphere", sphere);
        scene
            .world
            .get_mut(sphere)
            .unwrap()
            .move_by(nalgebra::Vector3::new(0.0, 0.0, -3.0));
        scene.add(
            TriangleMesh3D::box_mesh(
                nalgebra::Point3::new(0.0, 0.0, mesh_z),
                nalgebra::Vector3::new(0.8, 0.8, 0.05),
                Color::new(20, 255, 30, 255),
            )
            .with_material(SurfaceMaterial {
                base_color: [0.03, 0.9, 0.05, 1.0],
                emissive: [0.0, 1.0, 0.0],
                emissive_strength: 1.0,
                ..Default::default()
            }),
        );
        scene
    };
    let center =
        (config.output_height / 2 * config.output_width + config.output_width / 2) as usize;
    renderer.render_scene_with_outputs(
        &depth_scene(-4.5),
        &config,
        None,
        RenderOutputs::CPU_RGBA_ONLY,
    );
    let behind_pixel: [u8; 4] = renderer.get_rgba_bytes().unwrap()[center * 4..center * 4 + 4]
        .try_into()
        .unwrap();
    assert_eq!(renderer.last_stats().surface_lighting_dispatches, 1);
    assert!(
        behind_pixel[0] > behind_pixel[1],
        "SDF must occlude a raster mesh behind it"
    );

    renderer.render_scene_with_outputs(
        &depth_scene(-1.5),
        &config,
        None,
        RenderOutputs::CPU_RGBA_ONLY,
    );
    let front_pixel = &renderer.get_rgba_bytes().unwrap()[center * 4..center * 4 + 4];
    assert!(
        front_pixel[1] > front_pixel[0],
        "raster mesh must occlude an SDF behind it"
    );

    let mut single_sample_renderer = VulkanRenderer::new(
        context.clone(),
        RendererConfig {
            msaa_samples: 1,
            ssaa_factor: 1,
            output_color_profile: Default::default(),
            analytic_aa_2d: true,
        },
    );
    let (stats, non_black) =
        render_and_count_non_black(&mut single_sample_renderer, &depth_scene(-1.5), &config);
    assert_eq!(stats.surface_lighting_dispatches, 1);
    assert!(non_black > 100);

    let mut deferred_with_overlay = Scene::default();
    deferred_with_overlay.add(TriangleMesh3D::box_mesh(
        nalgebra::Point3::new(0.0, 0.0, -2.0),
        nalgebra::Vector3::new(0.8, 0.8, 0.8),
        Color::new(40, 120, 230, 255),
    ));
    let overlay = Rectangle {
        p0: nalgebra::Point3::new(-0.4, -0.1, 0.0),
        p1: nalgebra::Point3::new(0.4, -0.1, 0.0),
        p2: nalgebra::Point3::new(0.4, 0.1, 0.0),
        p3: nalgebra::Point3::new(-0.4, 0.1, 0.0),
        color: Color::new(255, 240, 30, 255),
        ..Default::default()
    };
    deferred_with_overlay.add_rectangle(overlay);
    let (stats, non_black) =
        render_and_count_non_black(&mut renderer, &deferred_with_overlay, &config);
    assert_eq!(stats.surface_lighting_dispatches, 1);
    assert_eq!(stats.mesh_2d_draw_calls, 1);
    assert_eq!(stats.surface_resolve_dispatches, 1);
    assert!(non_black > 100);

    let mut grid = Scene::default();
    grid.add(GridPlane3D::new(
        GridPlane::Xy,
        nalgebra::Point3::new(0.0, 0.0, -3.0),
        20.0,
        GridStyle3D {
            major_color: [0.0; 4],
            minor_color: [0.0; 4],
            u_axis_color: [1.0, 0.0, 0.0, 1.0],
            v_axis_color: [0.0, 1.0, 0.0, 1.0],
            line_width_pixels: 1.0,
            ..Default::default()
        },
    ));
    let grid_energy_ssaa_1 = render_coverage(&mut single_sample_renderer, &grid, &config);
    let mut ssaa_2_renderer = VulkanRenderer::new(
        context.clone(),
        RendererConfig {
            msaa_samples: 1,
            ssaa_factor: 2,
            output_color_profile: Default::default(),
            analytic_aa_2d: true,
        },
    );
    let grid_energy_ssaa_2 = render_coverage(&mut ssaa_2_renderer, &grid, &config);
    let mut ssaa_4_renderer = VulkanRenderer::new(
        context,
        RendererConfig {
            msaa_samples: 1,
            ssaa_factor: 4,
            output_color_profile: Default::default(),
            analytic_aa_2d: true,
        },
    );
    let grid_energy_ssaa_4 = render_coverage(&mut ssaa_4_renderer, &grid, &config);
    let grid_energy_ratio_2 = grid_energy_ssaa_2 as f64 / grid_energy_ssaa_1 as f64;
    let grid_energy_ratio = grid_energy_ssaa_4 as f64 / grid_energy_ssaa_1 as f64;
    assert!(
        (0.9..=1.1).contains(&grid_energy_ratio_2) && (0.9..=1.1).contains(&grid_energy_ratio),
        "procedural-grid coverage must stay stable across SSAA factors: \
         SSAA 1 energy={grid_energy_ssaa_1}, SSAA 2 energy={grid_energy_ssaa_2}, \
         SSAA 4 energy={grid_energy_ssaa_4}, ratios={grid_energy_ratio_2:.3}/{grid_energy_ratio:.3}"
    );

    let mut distant_grid = Scene::default();
    distant_grid.add(GridPlane3D::new(
        GridPlane::Xy,
        nalgebra::Point3::new(0.0, 0.0, -20.0),
        100.0,
        GridStyle3D {
            major_color: [1.0, 1.0, 1.0, 0.6],
            minor_color: [0.7, 0.7, 0.7, 0.3],
            u_axis_color: [0.0; 4],
            v_axis_color: [0.0; 4],
            cell_size: 1.0,
            subdivisions: 5,
            line_width_pixels: 1.0,
            ..Default::default()
        },
    ));
    let distant_energy_ssaa_1 =
        render_coverage(&mut single_sample_renderer, &distant_grid, &config);
    let distant_energy_ssaa_2 = render_coverage(&mut ssaa_2_renderer, &distant_grid, &config);
    let distant_energy_ssaa_4 = render_coverage(&mut ssaa_4_renderer, &distant_grid, &config);
    let distant_energy_ratio_2 = distant_energy_ssaa_2 as f64 / distant_energy_ssaa_1 as f64;
    let distant_energy_ratio = distant_energy_ssaa_4 as f64 / distant_energy_ssaa_1 as f64;
    assert!(
        (0.9..=1.1).contains(&distant_energy_ratio_2)
            && (0.9..=1.1).contains(&distant_energy_ratio),
        "procedural-grid Nyquist fading must stay stable across SSAA factors: \
         SSAA 1 energy={distant_energy_ssaa_1}, SSAA 2 energy={distant_energy_ssaa_2}, \
         SSAA 4 energy={distant_energy_ssaa_4}, \
         ratios={distant_energy_ratio_2:.3}/{distant_energy_ratio:.3}"
    );

    println!("render graph verification passed");
}
