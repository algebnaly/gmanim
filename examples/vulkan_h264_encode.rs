use std::io;
use std::process::Command;

use gmanim_core::{
    animation::{Animation, Timeline},
    math_utils::constants::PI,
    mobjects::{Rectangle, Transform},
    video_backend::{vulkan_h264::VulkanH264Backend, ColorOrder, VideoConfig},
    Color, Context, GMFloat, Scene, SceneConfig,
};

struct OrbitRects {
    frames: u32,
}

impl Animation for OrbitRects {
    fn update(&mut self, alpha: GMFloat, scene: &mut Scene) {
        let angle = alpha as f32 * PI * 2.0;
        for (i, m) in scene.mobjects.iter().enumerate() {
            let ring = (i % 6) as f32 + 1.0;
            let local = angle + i as f32 * 0.37;
            let x = local.cos() * ring * 0.55;
            let y = local.sin() * ring * 0.32;
            let matrix =
                nalgebra::Matrix4::new_translation(&nalgebra::Vector3::new(
                    x as GMFloat,
                    y as GMFloat,
                    0.0,
                )) * nalgebra::Matrix4::from_euler_angles(0.0, 0.0, -angle * 1.7 + i as f32 * 0.11);
            m.borrow_mut().set_model_matrix(matrix);
        }
    }

    fn total_frames(&self) -> u32 {
        self.frames
    }
}

fn main() -> io::Result<()> {
    let width = 1920u32;
    let height = 1080u32;
    let frames = 480u32;
    let output = "/tmp/gmanim_vulkan_h264_encode.mp4";

    let ctx = Context {
        scene_config: SceneConfig {
            width: 16.0,
            height: 9.0,
            output_width: width,
            output_height: height,
            scale_factor: height as GMFloat / 9.0,
            msaa_samples: 16,
            ssaa_factor: 1,
        },
    };

    let mut scene = Scene::default();
    for i in 0..96 {
        let mut rect = Rectangle {
            p0: nalgebra::Point3::new(-0.16, -0.16, 0.0),
            p1: nalgebra::Point3::new(0.16, -0.16, 0.0),
            p2: nalgebra::Point3::new(0.16, 0.16, 0.0),
            p3: nalgebra::Point3::new(-0.16, 0.16, 0.0),
            color: Color::new(
                (40 + (i * 37) % 200) as u8,
                (80 + (i * 17) % 160) as u8,
                (140 + (i * 29) % 110) as u8,
                255,
            ),
            ..Default::default()
        };
        rect.apply_transform(nalgebra::Matrix4::new_translation(&nalgebra::Vector3::new(
            ((i % 12) as f32 - 5.5) as GMFloat,
            ((i / 12) as f32 - 3.5) as GMFloat,
            0.0,
        )));
        rect.update_mesh();
        scene.add(rect);
    }

    let mut timeline = Timeline::new(scene, ctx);
    timeline.play(OrbitRects { frames });

    let video_config = VideoConfig {
        filename: output.to_owned(),
        framerate: 120,
        output_width: width,
        output_height: height,
        color_order: ColorOrder::Nv12,
        bitrate: Some(20_000_000),
    };
    let mut backend = pollster::block_on(VulkanH264Backend::try_new(&video_config))?;

    let start = std::time::Instant::now();
    let mut rendered = 0u32;
    while timeline.step_frame_for_vulkan_video() {
        let frame = timeline
            .vulkan_video_frame()
            .ok_or_else(|| io::Error::other("renderer did not produce a Vulkan video frame"))?;
        backend.submit_vulkan_frame(frame)?;
        rendered += 1;
    }
    backend.finish()?;

    let decoded = decode_rgba_frame(output, width, height, frames / 2)?;
    let non_black_pixels = decoded
        .chunks_exact(4)
        .filter(|px| px[0] > 8 || px[1] > 8 || px[2] > 8)
        .count();
    if non_black_pixels < (width * height / 40) as usize {
        return Err(io::Error::other(
            "decoded Vulkan H.264 MP4 frame appears to be empty",
        ));
    }

    println!(
        "encoded {rendered} frames to {output} in {:?}; decoded frame check passed",
        start.elapsed()
    );
    Ok(())
}

fn decode_rgba_frame(path: &str, width: u32, height: u32, frame_index: u32) -> io::Result<Vec<u8>> {
    let select = format!("select=eq(n\\,{frame_index}),format=rgba");
    let output = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            path,
            "-vf",
            &select,
            "-vframes",
            "1",
            "-f",
            "rawvideo",
            "-",
        ])
        .output()?;
    if !output.status.success() {
        return Err(io::Error::other(format!(
            "ffmpeg decode failed with status {}",
            output.status
        )));
    }
    let expected = (width * height * 4) as usize;
    if output.stdout.len() != expected {
        return Err(io::Error::other(format!(
            "decoded frame size mismatch: got {}, expected {expected}",
            output.stdout.len()
        )));
    }
    Ok(output.stdout)
}
