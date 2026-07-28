use gmanim_core::{
    Color, RendererConfig, Scene, SceneConfig,
    mobjects::{
        MobjectBase, Rectangle, Transform,
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

fn main() {
    let config = SceneConfig {
        width: 16.0,
        height: 9.0,
        output_width: 320,
        output_height: 180,
        scale_factor: 20.0,
    };
    let context = VulkanContext::new().unwrap();
    let mut renderer = VulkanRenderer::new(
        context.clone(),
        RendererConfig {
            msaa_samples: 4,
            ssaa_factor: 1,
        },
    );

    let empty = Scene::default();
    let (stats, non_black) = render_and_count_non_black(&mut renderer, &empty, &config);
    assert_eq!(stats.sdf_dispatches, 0);
    assert_eq!(stats.raster_passes, 0);
    assert_eq!(stats.surface_merge_dispatches, 0);
    assert_eq!(non_black, 0);

    let mut sdf = Scene::default();
    let mut sphere = Sphere3D {
        base: MobjectBase::new("verify-sphere"),
        radius: 1.0,
        material: SurfaceMaterial {
            base_color: [0.86, 0.31, 0.24, 1.0],
            ..Default::default()
        },
    };
    sphere.move_this(nalgebra::Vector3::new(0.0, 0.0, -3.0));
    sdf.add(sphere);
    let (stats, non_black) = render_and_count_non_black(&mut renderer, &sdf, &config);
    assert_eq!(stats.sdf_dispatches, 1);
    assert_eq!(stats.raster_passes, 0);
    assert_eq!(stats.surface_merge_dispatches, 1);
    assert!(non_black > 100);

    let mut raster = Scene::default();
    let mut rectangle = Rectangle {
        p0: nalgebra::Point3::new(-2.0, -1.0, 0.0),
        p1: nalgebra::Point3::new(2.0, -1.0, 0.0),
        p2: nalgebra::Point3::new(2.0, 1.0, 0.0),
        p3: nalgebra::Point3::new(-2.0, 1.0, 0.0),
        color: Color::new(40, 180, 240, 255),
        ..Default::default()
    };
    rectangle.update_mesh();
    raster.add(rectangle);
    let (stats, non_black) = render_and_count_non_black(&mut renderer, &raster, &config);
    assert_eq!(stats.sdf_dispatches, 0);
    assert_eq!(stats.raster_passes, 1);
    assert_eq!(stats.downsample_dispatches, 0);
    assert_eq!(stats.surface_merge_dispatches, 0);
    assert!(non_black > 100);

    let mut mixed = sdf;
    let mut overlay = Rectangle {
        p0: nalgebra::Point3::new(-0.7, -0.2, 0.0),
        p1: nalgebra::Point3::new(0.7, -0.2, 0.0),
        p2: nalgebra::Point3::new(0.7, 0.2, 0.0),
        p3: nalgebra::Point3::new(-0.7, 0.2, 0.0),
        color: Color::new(30, 220, 120, 255),
        ..Default::default()
    };
    overlay.update_mesh();
    mixed.add(overlay);
    let (stats, non_black) = render_and_count_non_black(&mut renderer, &mixed, &config);
    assert_eq!(stats.sdf_dispatches, 1);
    assert_eq!(stats.raster_passes, 1);
    assert_eq!(stats.surface_merge_dispatches, 1);
    assert!(non_black > 100);

    let depth_scene = |mesh_z| {
        let mut scene = Scene::default();
        let mut sphere = Sphere3D {
            base: MobjectBase::new("depth-sphere"),
            radius: 1.0,
            material: SurfaceMaterial {
                base_color: [0.9, 0.08, 0.03, 1.0],
                emissive: [1.0, 0.0, 0.0],
                emissive_strength: 1.0,
                ..Default::default()
            },
        };
        sphere.move_this(nalgebra::Vector3::new(0.0, 0.0, -3.0));
        scene.add(sphere);
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
    let behind_pixel = &renderer.get_rgba_bytes().unwrap()[center * 4..center * 4 + 4];
    assert_eq!(renderer.last_stats().raster_lighting_dispatches, 1);
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
        context,
        RendererConfig {
            msaa_samples: 1,
            ssaa_factor: 1,
        },
    );
    let (stats, non_black) =
        render_and_count_non_black(&mut single_sample_renderer, &depth_scene(-1.5), &config);
    assert_eq!(stats.raster_lighting_dispatches, 1);
    assert!(non_black > 100);

    let mut deferred_with_overlay = Scene::default();
    deferred_with_overlay.add(TriangleMesh3D::box_mesh(
        nalgebra::Point3::new(0.0, 0.0, -2.0),
        nalgebra::Vector3::new(0.8, 0.8, 0.8),
        Color::new(40, 120, 230, 255),
    ));
    let mut overlay = Rectangle {
        p0: nalgebra::Point3::new(-0.4, -0.1, 0.0),
        p1: nalgebra::Point3::new(0.4, -0.1, 0.0),
        p2: nalgebra::Point3::new(0.4, 0.1, 0.0),
        p3: nalgebra::Point3::new(-0.4, 0.1, 0.0),
        color: Color::new(255, 240, 30, 255),
        ..Default::default()
    };
    overlay.update_mesh();
    deferred_with_overlay.add(overlay);
    let (stats, non_black) =
        render_and_count_non_black(&mut renderer, &deferred_with_overlay, &config);
    assert_eq!(stats.raster_lighting_dispatches, 1);
    assert_eq!(stats.mesh_2d_draw_calls, 1);
    assert_eq!(stats.surface_merge_dispatches, 1);
    assert!(non_black > 100);

    println!("render graph verification passed");
}
