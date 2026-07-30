use gmanim_core::{
    Color, Context, GMFloat, Scene, SceneConfig,
    animation::{AnimationClip, Curve, TimelineBuilder},
    math_utils::constants::PI,
    mobjects::{MobjectId, Rectangle},
    video_backend::{ColorOrder, VideoBackend, VideoBackendType, VideoConfig},
};

fn rotation_clip(frames: u32, targets: &[MobjectId]) -> AnimationClip {
    let mut clip = AnimationClip::new(frames);
    for (i, id) in targets.iter().copied().enumerate() {
        let values: Vec<_> = (0..=frames)
            .map(|frame| {
                let angle = frame as GMFloat / frames as GMFloat * PI * 2.0;
                let x = (i as f32 % 40.0) - 20.0;
                let y = ((i / 40) as f32 % 25.0) - 12.5;
                nalgebra::Matrix4::new_translation(&nalgebra::Vector3::new(
                    x as GMFloat,
                    y as GMFloat,
                    0.0,
                )) * nalgebra::Matrix4::from_euler_angles(0.0, 0.0, angle + i as f32)
            })
            .collect();
        clip = clip.transform(id, Curve::sampled(values));
    }
    clip
}

fn main() {
    let width = 1920u32;
    let height = 1080u32;

    let ctx = Context {
        scene_config: SceneConfig {
            width: 16.0,
            height: 9.0,
            output_width: width,
            output_height: height,
            scale_factor: 1920.0 / 16.0,
            framerate: 60,
        },
    };

    let mut scene = Scene::default();
    let mut targets = Vec::with_capacity(1000);

    // Add 1000 rectangles
    for i in 0..1000 {
        let x = (i as f32 % 40.0) - 20.0;
        let y = ((i / 40) as f32 % 25.0) - 12.5;
        let rect = Rectangle {
            p0: nalgebra::Point3::new(-0.2, -0.2, 0.0),
            p1: nalgebra::Point3::new(0.2, -0.2, 0.0),
            p2: nalgebra::Point3::new(0.2, 0.2, 0.0),
            p3: nalgebra::Point3::new(-0.2, 0.2, 0.0),
            color: Color::new((i % 255) as u8, 100, 200, 255),
            ..Default::default()
        };
        let id = scene.add_rectangle(rect);
        targets.push(id);
        scene
            .world
            .get_mut(id)
            .unwrap()
            .move_by(nalgebra::Vector3::new(x as GMFloat, y as GMFloat, 0.0));
    }

    let mut timeline_builder = TimelineBuilder::new(scene, ctx);
    timeline_builder.play(rotation_clip(300, &targets)).unwrap();
    let mut timeline = timeline_builder.build();

    let video_config = VideoConfig {
        filename: "bench_core_nv12_2d.mp4".to_owned(),
        framerate: 60,
        output_width: width,
        output_height: height,
        color_order: ColorOrder::Nv12,
        bitrate: None,
        output_color_profile: Default::default(),
    };

    let mut video_backend = VideoBackend {
        backend_type: VideoBackendType::FfmpegPipe(
            gmanim_core::video_backend::FfmpegPipeBackend::new(
                &video_config,
                gmanim_core::video_backend::FfmpegPipeEncoder::HevcVaapi,
                false,
            ),
        ),
    };

    println!("Starting render loop...");
    let start_time = std::time::Instant::now();

    let mut frames_rendered = 0;
    let vk_ctx = gmanim_core::vulkan::context::VulkanContext::new().unwrap();
    let mut renderer = gmanim_core::vulkan::renderer::VulkanRenderer::new(
        vk_ctx,
        gmanim_core::RendererConfig {
            msaa_samples: 8,
            ssaa_factor: 2,
            output_color_profile: Default::default(),
        },
    );

    let mut first_frame_stats = None;
    let mut steady_frame_stats = None;
    while timeline.advance_frame().unwrap() {
        frames_rendered += 1;
        renderer.render_scene_with_outputs(
            &timeline.scene,
            &timeline.ctx.scene_config,
            None,
            gmanim_core::vulkan::renderer::RenderOutputs::CPU_NV12_ONLY,
        );
        let stats = renderer.last_stats();
        first_frame_stats.get_or_insert(stats);
        if frames_rendered == 2 {
            steady_frame_stats = Some(stats);
        }
        if let Some(nv12_bytes) = renderer.get_nv12_bytes() {
            let mut buf = video_backend.acquire_buffer();
            buf.as_mut_slice().copy_from_slice(nv12_bytes);
            video_backend.submit_frame(buf);
        }
    }

    println!("Frames rendered: {}", frames_rendered);
    println!("First frame renderer stats: {first_frame_stats:?}");
    println!("Steady frame renderer stats: {steady_frame_stats:?}");
    assert_eq!(first_frame_stats.unwrap().mesh_2d_geometry_uploads, 1);
    assert_eq!(steady_frame_stats.unwrap().mesh_2d_geometry_uploads, 0);
    assert_eq!(renderer.last_stats().mesh_2d_draw_calls, 1);
    assert_eq!(renderer.last_stats().mesh_2d_instances, 1000);
    assert_eq!(renderer.last_stats().sdf_dispatches, 0);
    assert_eq!(renderer.last_stats().raster_passes, 1);
    assert_eq!(renderer.last_stats().downsample_dispatches, 1);
    assert_eq!(renderer.last_stats().surface_resolve_dispatches, 0);
    assert_eq!(renderer.last_stats().surface_composite_dispatches, 0);
    drop(video_backend);

    println!("Total time: {:?}", start_time.elapsed());
}
