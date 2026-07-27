use gmanim_core::{
    Color, RendererConfig, Scene, SceneConfig,
    mobjects::{MobjectBase, Rectangle, Transform, object_3d::Sphere3D},
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
        context,
        RendererConfig {
            msaa_samples: 4,
            ssaa_factor: 1,
        },
    );

    let empty = Scene::default();
    let (stats, non_black) = render_and_count_non_black(&mut renderer, &empty, &config);
    assert_eq!(stats.sdf_dispatches, 0);
    assert_eq!(stats.raster_passes, 0);
    assert_eq!(stats.composite_dispatches, 0);
    assert_eq!(non_black, 0);

    let mut sdf = Scene::default();
    let mut sphere = Sphere3D {
        base: MobjectBase::new("verify-sphere"),
        radius: 1.0,
        color: Color::new(220, 80, 60, 255),
    };
    sphere.move_this(nalgebra::Vector3::new(0.0, 0.0, -3.0));
    sdf.add(sphere);
    let (stats, non_black) = render_and_count_non_black(&mut renderer, &sdf, &config);
    assert_eq!(stats.sdf_dispatches, 1);
    assert_eq!(stats.raster_passes, 0);
    assert_eq!(stats.composite_dispatches, 0);
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
    assert_eq!(stats.composite_dispatches, 0);
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
    assert_eq!(stats.composite_dispatches, 1);
    assert!(non_black > 100);

    println!("render graph verification passed");
}
