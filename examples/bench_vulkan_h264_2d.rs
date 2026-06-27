use gmanim_core::{
    animation::{Animation, Timeline},
    math_utils::constants::PI,
    mobjects::{Rectangle, Transform},
    video_backend::{vulkan_h264::VulkanH264Backend, ColorOrder, VideoConfig},
    Color, Context, GMFloat, Scene, SceneConfig,
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

fn main() -> std::io::Result<()> {
    let width = 1920u32;
    let height = 1080u32;
    let frames = 300u32;

    let ctx = Context {
        scene_config: SceneConfig {
            width: 16.0,
            height: 9.0,
            output_width: width,
            output_height: height,
            scale_factor: height as GMFloat / 9.0,
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
        filename: "/tmp/gmanim-bench-vulkan-h264-2d.mp4".to_owned(),
        framerate: 60,
        output_width: width,
        output_height: height,
        color_order: ColorOrder::Nv12,
    };
    let mut video_backend = pollster::block_on(VulkanH264Backend::try_new(&video_config))?;

    println!("Starting Vulkan H.264 render loop...");
    let start_time = std::time::Instant::now();

    let mut frames_rendered = 0;
    while timeline.step_frame_for_vulkan_video() {
        frames_rendered += 1;
        let frame = timeline
            .vulkan_video_frame()
            .ok_or_else(|| std::io::Error::other("missing Vulkan video frame"))?;
        video_backend.submit_vulkan_frame(frame)?;
    }
    video_backend.finish()?;

    println!("Frames rendered: {}", frames_rendered);
    println!("Total time: {:?}", start_time.elapsed());
    Ok(())
}
