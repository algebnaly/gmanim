use gmanim_core::{
    Color, Context, GMFloat, Scene, SceneConfig,
    animation::{Animation, Timeline},
    math_utils::constants::PI,
    mobjects::{Rectangle, Transform},
    video_backend::{ColorOrder, VideoBackend, VideoBackendType, VideoConfig},
};

struct RotateRectangles {
    frames: u32,
}

impl Animation for RotateRectangles {
    fn update(&mut self, alpha: GMFloat, scene: &mut Scene) {
        let angle = alpha as f32 * PI * 2.0;
        for (i, m) in scene.mobjects.iter().enumerate() {
            let x = (i as f32 % 40.0) - 20.0;
            let y = ((i / 40) as f32 % 25.0) - 12.5;
            let matrix = nalgebra::Matrix4::new_translation(&nalgebra::Vector3::new(
                x as GMFloat,
                y as GMFloat,
                0.0,
            )) * nalgebra::Matrix4::from_euler_angles(0.0, 0.0, angle + i as f32);
            m.borrow_mut().set_model_matrix(matrix);
        }
    }
    fn total_frames(&self) -> u32 {
        self.frames
    }
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
        },
    };

    let mut scene = Scene::default();

    // Add 1000 rectangles
    for i in 0..1000 {
        let x = (i as f32 % 40.0) - 20.0;
        let y = ((i / 40) as f32 % 25.0) - 12.5;
        let mut rect = Rectangle {
            p0: nalgebra::Point3::new(-0.2, -0.2, 0.0),
            p1: nalgebra::Point3::new(0.2, -0.2, 0.0),
            p2: nalgebra::Point3::new(0.2, 0.2, 0.0),
            p3: nalgebra::Point3::new(-0.2, 0.2, 0.0),
            color: Color::new((i % 255) as u8, 100, 200, 255),
            ..Default::default()
        };
        rect.apply_transform(nalgebra::Matrix4::new_translation(&nalgebra::Vector3::new(
            x as GMFloat,
            y as GMFloat,
            0.0,
        )));
        rect.update_mesh();
        scene.add(rect);
    }

    let mut timeline = Timeline::new(scene, ctx);
    timeline.play(RotateRectangles { frames: 300 });

    let video_config = VideoConfig {
        filename: "bench_core_nv12_2d.mp4".to_owned(),
        framerate: 60,
        output_width: width,
        output_height: height,
        color_order: ColorOrder::Nv12,
        bitrate: None,
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
        },
    );

    let mut first_frame_stats = None;
    let mut steady_frame_stats = None;
    while timeline.advance_frame() {
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
    assert_eq!(renderer.last_stats().composite_dispatches, 0);
    drop(video_backend);

    println!("Total time: {:?}", start_time.elapsed());
}
