use gmanim_core::{
    Color, Context, GMFloat, Scene, SceneConfig,
    animation::{Animation, Timeline},
    math_utils::constants::PI,
    mobjects::{Rectangle, Transform},
    video_backend::{
        ColorOrder, VideoConfig,
        vulkan_h264::{AsyncVulkanH264Backend, H264RateControlPolicy, VulkanH264EncoderConfig},
    },
    vulkan::renderer::GpuPassTimings,
};

struct RotateRectangles {
    frames: u32,
}

#[derive(Default)]
struct GpuTimingAccumulator {
    sum: GpuPassTimings,
    samples: u32,
}

impl GpuTimingAccumulator {
    fn push(&mut self, timings: GpuPassTimings) {
        self.sum.frame_ms += timings.frame_ms;
        self.sum.geometry_upload_ms += timings.geometry_upload_ms;
        self.sum.sdf_ms += timings.sdf_ms;
        self.sum.raster_ms += timings.raster_ms;
        self.sum.postprocess_ms += timings.postprocess_ms;
        self.sum.output_ms += timings.output_ms;
        self.samples += 1;
    }

    fn average(&self) -> GpuPassTimings {
        let divisor = self.samples.max(1) as f64;
        GpuPassTimings {
            frame_ms: self.sum.frame_ms / divisor,
            geometry_upload_ms: self.sum.geometry_upload_ms / divisor,
            sdf_ms: self.sum.sdf_ms / divisor,
            raster_ms: self.sum.raster_ms / divisor,
            postprocess_ms: self.sum.postprocess_ms / divisor,
            output_ms: self.sum.output_ms / divisor,
        }
    }
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

fn main() -> std::io::Result<()> {
    let width = 1920u32;
    let height = 1080u32;
    let frames = std::env::var("GMANIM_BENCH_FRAMES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(300u32);
    let use_p_frames = std::env::var("GMANIM_H264_P_FRAMES")
        .map(|value| value != "0")
        .unwrap_or(true);
    let rate_control_env = std::env::var("GMANIM_H264_RATE_CONTROL");
    let rate_control = match rate_control_env.as_deref() {
        Ok("vbr") | Err(std::env::VarError::NotPresent) => H264RateControlPolicy::Vbr,
        Ok("cbr") => H264RateControlPolicy::Cbr,
        Ok("disabled") => H264RateControlPolicy::Disabled,
        Ok(value) => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "invalid GMANIM_H264_RATE_CONTROL={value:?}; expected vbr, cbr, or disabled"
                ),
            ));
        }
        Err(err) => return Err(std::io::Error::other(err.to_string())),
    };
    let output_filename = std::env::var("GMANIM_BENCH_OUTPUT")
        .unwrap_or_else(|_| "/tmp/gmanim-bench-vulkan-h264-2d.mp4".to_owned());
    let encoder_config = VulkanH264EncoderConfig {
        use_p_frames,
        gop_size: 60,
        rate_control,
    };

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
    timeline.play(RotateRectangles { frames });

    let video_config = VideoConfig {
        filename: output_filename.clone(),
        framerate: 60,
        output_width: width,
        output_height: height,
        color_order: ColorOrder::Nv12,
        bitrate: None,
    };
    let vk_ctx = gmanim_core::vulkan::context::VulkanContext::new().unwrap();
    let mut video_backend = AsyncVulkanH264Backend::try_new_with_encoder_config(
        vk_ctx.clone(),
        &video_config,
        encoder_config,
    )?;

    println!("Starting Vulkan H.264 render loop with {encoder_config:?}...");
    let start_time = std::time::Instant::now();

    let mut frames_rendered = 0;
    let mut renderer = gmanim_core::vulkan::renderer::VulkanRenderer::new(
        vk_ctx,
        gmanim_core::RendererConfig {
            msaa_samples: 8,
            ssaa_factor: 2,
        },
    );
    renderer.set_gpu_profiling(true);

    let mut first_frame_stats = None;
    let mut steady_frame_stats = None;
    let mut gpu_timings = GpuTimingAccumulator::default();
    let mut scene_update_time = std::time::Duration::ZERO;
    let mut render_submit_time = std::time::Duration::ZERO;
    let mut video_submit_time = std::time::Duration::ZERO;
    loop {
        let scene_update_start = std::time::Instant::now();
        if !timeline.advance_frame() {
            break;
        }
        scene_update_time += scene_update_start.elapsed();
        frames_rendered += 1;
        let render_submit_start = std::time::Instant::now();
        renderer.render_scene_with_outputs(
            &timeline.scene,
            &timeline.ctx.scene_config,
            None,
            gmanim_core::vulkan::renderer::RenderOutputs::VULKAN_VIDEO_ONLY,
        );
        render_submit_time += render_submit_start.elapsed();
        let stats = renderer.last_stats();
        first_frame_stats.get_or_insert(stats);
        if frames_rendered == 2 {
            steady_frame_stats = Some(stats);
        }
        if let Some(timings) = renderer.last_gpu_timings() {
            gpu_timings.push(timings);
        }
        let frame = renderer
            .get_vulkan_video_frame()
            .ok_or_else(|| std::io::Error::other("missing Vulkan video frame"))?;
        let video_submit_start = std::time::Instant::now();
        video_backend.submit_vulkan_frame(frame)?;
        video_submit_time += video_submit_start.elapsed();
    }
    let finish_start = std::time::Instant::now();
    video_backend.finish()?;
    let finish_time = finish_start.elapsed();
    let encode_stats = video_backend.stats();
    let output_bytes = std::fs::metadata(&output_filename)?.len();

    println!("Frames rendered: {}", frames_rendered);
    println!("First frame renderer stats: {first_frame_stats:?}");
    println!("Steady frame renderer stats: {steady_frame_stats:?}");
    println!(
        "Average GPU timings ({} samples): {:?}",
        gpu_timings.samples,
        gpu_timings.average()
    );
    let frame_divisor = frames_rendered.max(1);
    println!(
        "Average CPU scene update: {:?}",
        scene_update_time / frame_divisor
    );
    println!(
        "Average CPU render submission: {:?}",
        render_submit_time / frame_divisor
    );
    println!(
        "Average CPU video submission: {:?}",
        video_submit_time / frame_divisor
    );
    println!("Video finish and mux drain: {finish_time:?}");
    println!("Vulkan H.264 stats: {encode_stats:?}");
    println!(
        "Average encode completion interval: {:?}",
        encode_stats.average_completion_interval()
    );
    println!("Output bytes: {output_bytes}");
    assert_eq!(first_frame_stats.unwrap().mesh_2d_geometry_uploads, 1);
    assert_eq!(steady_frame_stats.unwrap().mesh_2d_geometry_uploads, 0);
    assert_eq!(renderer.last_stats().mesh_2d_draw_calls, 1);
    assert_eq!(renderer.last_stats().mesh_2d_instances, 1000);
    assert_eq!(renderer.last_stats().sdf_dispatches, 0);
    assert_eq!(renderer.last_stats().raster_passes, 1);
    assert_eq!(renderer.last_stats().depth_attachment_raster_passes, 0);
    assert_eq!(renderer.last_stats().downsample_dispatches, 0);
    assert_eq!(renderer.last_stats().fused_video_downsample_dispatches, 1);
    assert_eq!(renderer.last_stats().surface_resolve_dispatches, 0);
    println!("Total time: {:?}", start_time.elapsed());
    Ok(())
}
