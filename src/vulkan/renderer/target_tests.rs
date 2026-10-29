//! Explicit GPU regressions: cargo test --lib target_tests -- --ignored --test-threads=1
use super::*;
use crate::mobjects::{Rectangle, mesh_3d::TriangleMesh3D, object_3d::Sphere3D};
use crate::{Color, RendererConfig, Scene, SceneConfig};
use nalgebra::{Point3, Vector3};

fn config() -> SceneConfig {
    SceneConfig {
        width: 16.0,
        height: 9.0,
        output_width: 320,
        output_height: 180,
        scale_factor: 20.0,
        framerate: 60,
    }
}

fn rectangle_scene() -> Scene {
    let mut scene = Scene::default();
    let corners = [
        Point3::new(-2.0, -1.0, 0.0),
        Point3::new(2.0, -1.0, 0.0),
        Point3::new(2.0, 1.0, 0.0),
        Point3::new(-2.0, 1.0, 0.0),
    ];
    let id = scene.add_rectangle(Rectangle {
        p0: corners[0],
        p1: corners[1],
        p2: corners[2],
        p3: corners[3],
        color: Color::new(40, 180, 240, 255),
        draw_config: crate::mobjects::DrawConfig {
            stoke_width: 0.0,
            ..Default::default()
        },
        ..Default::default()
    });
    // Exercise dynamic geometry offsets across cache rebuilds at nonzero frame slots.
    scene.world.set_rectangle_corners(id, corners).unwrap();
    scene
}

fn assert_cpu_outputs(renderer: &mut VulkanRenderer, config: &SceneConfig, outputs: RenderOutputs) {
    let pixels = (config.output_width * config.output_height) as usize;
    let rgba = renderer.get_rgba_bytes();
    assert_eq!(rgba.is_some(), outputs.cpu_rgba);
    if let Some(bytes) = rgba {
        assert_eq!(bytes.len(), pixels * 4);
        assert!(bytes.chunks_exact(4).any(|pixel| pixel[3] != 0));
    }
    let nv12 = renderer.get_nv12_bytes();
    assert_eq!(nv12.is_some(), outputs.cpu_nv12);
    if let Some(bytes) = nv12 {
        assert_eq!(bytes.len(), pixels * 3 / 2);
        assert!(bytes[..pixels].iter().any(|&y| y > 16));
    }
    let yuv = renderer.get_yuv444p_bytes();
    assert_eq!(yuv.is_some(), outputs.cpu_yuv444p);
    if let Some(bytes) = yuv {
        assert_eq!(bytes.len(), pixels * 3);
        assert!(bytes[..pixels].iter().any(|&y| y > 16));
    }
}

fn assert_2d_targets(cache: &TargetCache, width: u32, height: u32) {
    assert!(!cache.requirements.sdf && !cache.requirements.surface);
    assert!(cache.msaa_texture.is_none());
    assert!(cache.msaa_depth_texture.is_none());
    assert!(cache.video_nv12_slots.is_empty());
    for targets in &cache.render_targets {
        assert_eq!(
            (
                targets.resolved_texture.width,
                targets.resolved_texture.height
            ),
            (width, height)
        );
        for image in [
            &targets.sdf_normal_coverage,
            &targets.sdf_material_id,
            &targets.sdf_depth,
            &targets.resolved_primary_normal_depth,
            &targets.resolved_primary_albedo_coverage,
            &targets.resolved_secondary_normal_depth,
            &targets.resolved_secondary_albedo_coverage,
            &targets.resolved_material_ids,
            &targets.surface_hdr,
            &targets.overlay_hdr,
            &targets.scene_color,
            &targets.transparent_back_depth,
            &targets.bloom_ping,
            &targets.bloom_pong,
        ] {
            assert_eq!((image.width, image.height), (1, 1));
        }
    }
}

fn target_image_bytes(cache: &TargetCache) -> u64 {
    let mut images = vec![
        &cache.raster_normal_depth,
        &cache.raster_albedo,
        &cache.raster_material_id,
    ];
    images.extend(cache.msaa_texture.iter());
    images.extend(cache.msaa_depth_texture.iter());
    for targets in &cache.render_targets {
        images.extend([
            &targets.texture,
            &targets.resolved_texture,
            &targets.sdf_normal_coverage,
            &targets.sdf_material_id,
            &targets.sdf_depth,
            &targets.resolved_primary_normal_depth,
            &targets.resolved_primary_albedo_coverage,
            &targets.resolved_secondary_normal_depth,
            &targets.resolved_secondary_albedo_coverage,
            &targets.resolved_material_ids,
            &targets.surface_hdr,
            &targets.overlay_hdr,
            &targets.scene_color,
            &targets.transparent_back_depth,
            &targets.bloom_ping,
            &targets.bloom_pong,
        ]);
    }
    images
        .iter()
        .map(|image| image.allocation.as_ref().unwrap().size())
        .sum()
}

#[test]
#[ignore = "requires a Vulkan graphics/compute device"]
fn targets_follow_scene_outputs_and_resize_without_stale_readbacks() {
    let ctx = VulkanContext::new().unwrap();
    let mut renderer = VulkanRenderer::new(
        ctx,
        RendererConfig {
            msaa_samples: 4,
            ssaa_factor: 2,
            ..Default::default()
        },
    );
    let scene = rectangle_scene();
    let mut config = config();
    let yuv_only = RenderOutputs {
        cpu_yuv444p: true,
        ..RenderOutputs::NONE
    };
    for outputs in [
        RenderOutputs::CPU_RGBA_ONLY,
        RenderOutputs::CPU_NV12_ONLY,
        yuv_only,
        RenderOutputs::CPU_READBACKS,
        RenderOutputs::NONE,
        RenderOutputs::CPU_RGBA_ONLY,
    ] {
        for _ in 0..4 {
            let previous_frame = renderer
                .cache
                .as_ref()
                .map_or(0, |cache| cache.current_frame);
            renderer.render_scene_with_outputs(&scene, &config, None, outputs);
            let cache = renderer.cache.as_ref().unwrap();
            assert_eq!(cache.current_frame, previous_frame + 1);
            assert_2d_targets(cache, config.output_width, config.output_height);
            assert_eq!(
                cache.output_buffers.len(),
                usize::from(outputs.cpu_rgba) * RENDER_FRAME_COUNT
            );
            assert_eq!(
                cache.nv12_output_buffers.len(),
                usize::from(outputs.cpu_nv12) * RENDER_FRAME_COUNT
            );
            assert_eq!(
                cache.yuv444p_output_buffers.len(),
                usize::from(outputs.cpu_yuv444p) * RENDER_FRAME_COUNT
            );
            assert_cpu_outputs(&mut renderer, &config, outputs);
        }
    }
    let stable_image = renderer.cache.as_ref().unwrap().render_targets[0]
        .texture
        .vk_image;
    let stable_composite_set = renderer.cache.as_ref().unwrap().render_targets[0]
        .composite_descriptor_set;
    renderer.render_scene_with_outputs(&scene, &config, None, RenderOutputs::CPU_RGBA_ONLY);
    assert_eq!(
        renderer.cache.as_ref().unwrap().render_targets[0]
            .texture
            .vk_image,
        stable_image
    );
    let baseline = renderer.get_rgba_bytes().unwrap().to_vec();

    let mut opaque = Scene::default();
    opaque.add(TriangleMesh3D::box_mesh(
        Point3::new(0.0, 0.0, -3.0),
        Vector3::repeat(0.5),
        Color::white(),
    ));
    renderer.render_scene_with_outputs(&opaque, &config, None, RenderOutputs::CPU_RGBA_ONLY);
    let cache = renderer.cache.as_ref().unwrap();
    // Incremental TargetCache::reconfigure preserves unchanged images and core descriptor sets.
    assert_eq!(cache.render_targets[0].texture.vk_image, stable_image);
    assert_eq!(
        cache.render_targets[0].composite_descriptor_set,
        stable_composite_set
    );
    assert!(cache.requirements.raster_gbuffer && cache.requirements.surface);
    assert!(cache.msaa_texture.is_none());
    assert!(cache.msaa_depth_texture.is_some());
    assert_eq!(cache.render_targets[0].sdf_depth.width, 1);
    assert_eq!(
        cache.render_targets[0].surface_hdr.width,
        config.output_width * 2
    );
    assert_cpu_outputs(&mut renderer, &config, RenderOutputs::CPU_RGBA_ONLY);

    let mut mixed = rectangle_scene();
    let sphere = mixed.add(Sphere3D {
        radius: 1.0,
        material: SurfaceMaterial::default(),
    });
    mixed
        .world
        .get_mut(sphere)
        .unwrap()
        .move_by(Vector3::new(0.0, 0.0, -3.0));
    mixed.add(TriangleMesh3D::box_mesh(
        Point3::new(0.0, 0.0, -4.0),
        Vector3::repeat(0.5),
        Color::white(),
    ));
    mixed.add(TriangleMesh3D::box_mesh(
        Point3::new(1.0, 0.0, -2.0),
        Vector3::repeat(0.5),
        Color::new(50, 150, 200, 100),
    ));
    renderer.render_scene_with_outputs(&mixed, &config, None, RenderOutputs::CPU_RGBA_ONLY);
    let cache = renderer.cache.as_ref().unwrap();
    assert!(cache.requirements.sdf && cache.requirements.surface && cache.requirements.transparent);
    assert_eq!(
        cache.render_targets[0].sdf_depth.width,
        config.output_width * 2
    );
    assert_eq!(
        cache.render_targets[0].scene_color.width,
        config.output_width * 2
    );
    assert_cpu_outputs(&mut renderer, &config, RenderOutputs::CPU_RGBA_ONLY);

    renderer.set_bloom_enabled(true);
    renderer.render_scene_with_outputs(&scene, &config, None, RenderOutputs::CPU_RGBA_ONLY);
    let cache = renderer.cache.as_ref().unwrap();
    assert_eq!(
        cache.render_targets[0].bloom_ping.width,
        config.output_width / 2
    );
    assert_eq!(
        cache.render_targets[0].resolved_texture.width,
        config.output_width * 2
    );
    assert_cpu_outputs(&mut renderer, &config, RenderOutputs::CPU_RGBA_ONLY);
    renderer.set_bloom_enabled(false);
    renderer.render_scene_with_outputs(&scene, &config, None, RenderOutputs::CPU_RGBA_ONLY);
    assert_2d_targets(
        renderer.cache.as_ref().unwrap(),
        config.output_width,
        config.output_height,
    );
    assert_eq!(renderer.get_rgba_bytes().unwrap(), baseline);

    config.output_width = 192;
    config.output_height = 108;
    config.scale_factor = 12.0;
    let mut explicit_output = vec![0; (config.output_width * config.output_height * 4) as usize];
    renderer.render_scene_with_outputs(
        &scene,
        &config,
        Some(&mut explicit_output),
        RenderOutputs::NONE,
    );
    assert_2d_targets(
        renderer.cache.as_ref().unwrap(),
        config.output_width,
        config.output_height,
    );
    assert_eq!(renderer.get_rgba_bytes().unwrap(), explicit_output);
    renderer.render_scene_with_outputs(
        &Scene::default(),
        &config,
        None,
        RenderOutputs::CPU_RGBA_ONLY,
    );
    assert_eq!(
        renderer.cache.as_ref().unwrap().render_targets[0]
            .resolved_texture
            .width,
        1
    );
    assert!(
        renderer
            .get_rgba_bytes()
            .unwrap()
            .iter()
            .all(|&byte| byte == 0)
    );

    config.output_width = 1920;
    config.output_height = 1080;
    config.scale_factor = 120.0;
    renderer.render_scene_with_outputs(&scene, &config, None, RenderOutputs::NONE);
    let cache = renderer.cache.as_ref().unwrap();
    assert_2d_targets(cache, 1920, 1080);
    let image_bytes = target_image_bytes(cache);
    eprintln!(
        "1080p analytic 2D, configured SSAA 2: target image allocations = {:.2} MiB",
        image_bytes as f64 / 1048576.0
    );
    // The old SDF + resolved-surface + surface-HDR images alone consumed ~1.39 GiB.
    assert!(image_bytes < 128 * 1024 * 1024);
    renderer.wait_idle();
}

#[test]
#[ignore = "requires Vulkan video encode image support"]
fn video_frame_owns_its_image_across_target_rebuilds() {
    let ctx = VulkanContext::new().unwrap();
    let mut renderer = VulkanRenderer::new(
        ctx,
        RendererConfig {
            msaa_samples: 1,
            ssaa_factor: 2,
            analytic_aa_2d: false,
            ..Default::default()
        },
    );
    let config = config();
    let scene = rectangle_scene();
    renderer.render_scene_with_outputs(&scene, &config, None, RenderOutputs::VULKAN_VIDEO_ONLY);
    assert_eq!(renderer.last_stats().fused_video_downsample_dispatches, 1);
    assert!(renderer.current_output_image_view().is_none());
    let cache = renderer.cache.as_ref().unwrap();
    assert!(
        cache.output_buffers.is_empty()
            && cache.nv12_output_buffers.is_empty()
            && cache.yuv444p_output_buffers.is_empty()
    );
    assert_eq!(cache.render_targets[0].texture.width, 1);
    assert_eq!(
        cache.video_nv12_slots.len(),
        video_output::VIDEO_NV12_IMAGE_COUNT
    );
    let old_image = Arc::downgrade(&cache.video_nv12_slots[0].image);
    let frame = renderer.get_vulkan_video_frame().unwrap();
    renderer.render_scene_with_outputs(&scene, &config, None, RenderOutputs::CPU_RGBA_ONLY);
    assert!(renderer.cache.as_ref().unwrap().video_nv12_slots.is_empty());
    assert!(old_image.upgrade().is_some());
    drop(frame);
    assert!(old_image.upgrade().is_none());
    assert_cpu_outputs(&mut renderer, &config, RenderOutputs::CPU_RGBA_ONLY);
}
