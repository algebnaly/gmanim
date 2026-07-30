use std::fs::File;
use std::io::{BufWriter, Write};

use gmanim_core::{
    Color, RendererConfig, Scene, SceneConfig,
    mobjects::mesh_3d::{
        AlphaMode3D, SphericalGridMaterial, SphericalPatchMaterial, SurfaceMaterial,
        Transmission3D, TriangleMesh3D,
    },
    vulkan::{
        context::VulkanContext,
        renderer::{RenderOutputs, VulkanRenderer},
    },
};
use nalgebra::{Point3, Vector3};

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
const SPHERE_RADIUS: f32 = 2.25;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = VulkanContext::new()?;
    let mut renderer = VulkanRenderer::new(
        context,
        RendererConfig {
            msaa_samples: 4,
            ssaa_factor: 1,
            output_color_profile: Default::default(),
        },
    );
    renderer.set_bloom_enabled(true);
    let config = SceneConfig {
        width: 16.0,
        height: 9.0,
        output_width: WIDTH,
        output_height: HEIGHT,
        scale_factor: WIDTH as f32 / 16.0,
        framerate: 60,
    };

    render_scene(
        &mut renderer,
        &opaque_reference_scene(),
        &config,
        "/tmp/gmanim_spherical_opaque.ppm",
    )?;
    render_scene(
        &mut renderer,
        &transparent_reference_scene(),
        &config,
        "/tmp/gmanim_spherical_transparent.ppm",
    )?;
    let stats = renderer.last_stats();
    assert_eq!(stats.mesh_3d_opaque_draw_calls, 4);
    assert_eq!(stats.mesh_3d_transparent_draw_calls, 2);
    assert_eq!(stats.surface_lighting_dispatches, 1);
    assert_eq!(stats.surface_resolve_dispatches, 1);
    assert_eq!(stats.surface_composite_dispatches, 1);
    assert_eq!(stats.tone_map_dispatches, 1);
    assert_eq!(stats.bloom_dispatches, 3);

    println!("wrote /tmp/gmanim_spherical_opaque.ppm");
    println!("wrote /tmp/gmanim_spherical_transparent.ppm");
    Ok(())
}

fn base_scene() -> Scene {
    let mut scene = Scene::default();
    scene.camera.position = Point3::new(0.0, 0.15, 7.2);
    scene.camera.set_look_at(Vector3::new(0.0, -0.02, -1.0));
    scene.camera.set_perspective(
        38.0_f32.to_radians(),
        WIDTH as f32 / HEIGHT as f32,
        0.1,
        100.0,
    );
    scene.point_light.position = Point3::new(-3.5, 4.0, 5.0);
    scene.point_light.color = Color::new(235, 245, 255, 255);
    scene.point_light.intensity = 720.0;
    scene.environment_light.color = Color::new(105, 130, 145, 255);
    scene.environment_light.intensity = 0.24;
    scene.environment_light.rotation_radians = -0.35;
    scene
}

fn opaque_reference_scene() -> Scene {
    let mut scene = base_scene();
    let sphere = TriangleMesh3D::uv_sphere(
        Point3::origin(),
        SPHERE_RADIUS,
        128,
        64,
        Color::new(102, 107, 108, 255),
    )
    .with_material(SurfaceMaterial {
        roughness: 0.78,
        metallic: 0.0,
        reflectance: 0.45,
        spherical_grid: Some(SphericalGridMaterial {
            color: [0.48, 0.7, 0.72, 0.72],
            longitude_count: 16.0,
            latitude_count: 12.0,
            line_width_pixels: 1.15,
            backface_intensity: 0.0,
        }),
        ..Default::default()
    });
    scene.add(sphere);
    scene
}

fn transparent_reference_scene() -> Scene {
    let mut scene = base_scene();
    scene.camera.position = Point3::new(0.25, 0.1, 7.4);
    scene.point_light.position = Point3::new(-3.5, 4.5, 4.0);
    let corners = [
        Vector3::new(-0.38, 0.78, 0.5).normalize(),
        Vector3::new(-0.99, -0.08, 0.08).normalize(),
        Vector3::new(-0.43, -0.78, 0.45).normalize(),
    ];

    let shell = TriangleMesh3D::uv_sphere(
        Point3::origin(),
        SPHERE_RADIUS,
        128,
        64,
        Color::new(72, 126, 151, 255),
    )
    .with_material(SurfaceMaterial {
        roughness: 0.18,
        metallic: 0.0,
        reflectance: 0.5,
        alpha_mode: AlphaMode3D::Blend(Transmission3D {
            opacity: 0.035,
            fresnel_opacity: 0.72,
            absorption: [0.38, 0.14, 0.045],
            ior: 1.44,
            backface_opacity_scale: 0.58,
        }),
        spherical_grid: Some(SphericalGridMaterial {
            color: [0.56, 0.82, 0.9, 0.62],
            longitude_count: 24.0,
            latitude_count: 16.0,
            line_width_pixels: 0.9,
            backface_intensity: 0.3,
        }),
        spherical_patch: Some(SphericalPatchMaterial {
            directions: corners.map(|corner| [corner.x, corner.y, corner.z]),
            color: [0.2, 0.67, 0.82, 0.68],
            edge_color: [1.0, 0.84, 0.03, 1.0],
            edge_width_pixels: 2.2,
        }),
        ..Default::default()
    });
    scene.add(shell);

    let marker_material = SurfaceMaterial {
        emissive: [1.0, 0.84, 0.02],
        emissive_strength: 0.7,
        roughness: 0.22,
        metallic: 0.05,
        reflectance: 0.62,
        ..Default::default()
    };
    for corner in corners {
        scene.add(
            TriangleMesh3D::uv_sphere(
                Point3::from(corner * (SPHERE_RADIUS * 1.025)),
                0.07,
                24,
                12,
                Color::new(255, 224, 20, 255),
            )
            .with_material(marker_material),
        );
    }
    scene.add(
        TriangleMesh3D::uv_sphere(
            Point3::new(0.0, -0.05, 0.35),
            0.11,
            32,
            16,
            Color::new(255, 58, 42, 255),
        )
        .with_material(SurfaceMaterial {
            emissive: [1.0, 0.03, 0.01],
            emissive_strength: 1.5,
            roughness: 0.2,
            metallic: 0.0,
            reflectance: 0.5,
            ..Default::default()
        }),
    );
    scene
}

fn render_scene(
    renderer: &mut VulkanRenderer,
    scene: &Scene,
    config: &SceneConfig,
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut rgba = vec![0_u8; (WIDTH * HEIGHT * 4) as usize];
    renderer.render_scene_with_outputs(
        scene,
        config,
        Some(&mut rgba),
        RenderOutputs::CPU_RGBA_ONLY,
    );
    assert_visible_frame(path, &rgba);
    write_ppm(path, WIDTH, HEIGHT, &rgba)?;
    Ok(())
}

fn assert_visible_frame(path: &str, rgba: &[u8]) {
    let visible_pixels = rgba
        .chunks_exact(4)
        .filter(|pixel| pixel[..3].iter().copied().max().unwrap_or(0) > 8)
        .count();
    let rgb_energy: u64 = rgba
        .chunks_exact(4)
        .flat_map(|pixel| pixel[..3].iter().copied())
        .map(u64::from)
        .sum();
    let average_rgb = rgb_energy as f64 / (WIDTH as f64 * HEIGHT as f64 * 3.0);

    assert!(
        visible_pixels > 100_000,
        "{path} lost visible geometry: only {visible_pixels} lit pixels"
    );
    assert!(
        average_rgb > 5.0,
        "{path} is effectively black: average RGB energy is {average_rgb:.3}"
    );
}

fn write_ppm(path: &str, width: u32, height: u32, rgba: &[u8]) -> std::io::Result<()> {
    let mut output = BufWriter::new(File::create(path)?);
    write!(output, "P6\n{width} {height}\n255\n")?;
    for pixel in rgba.chunks_exact(4) {
        output.write_all(&pixel[..3])?;
    }
    output.flush()
}
