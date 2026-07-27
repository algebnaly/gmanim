use std::collections::VecDeque;
use std::fmt::Display;
use std::io;
use std::sync::mpsc::{self, Receiver, Sender, channel};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

use crate::video_backend::ffmpeg::FfmpegBackend;
use crate::video_backend::vaapi::FfmpegVaapiBackend;
pub mod ffmpeg;
pub mod vaapi;
pub mod vulkan_h264;

const BLOCK_SIZE: usize = 240;
pub enum VideoBackendType {
    FfmpegPipe(FfmpegPipeBackend),
    Ffmpeg(FfmpegBackend),
    Vaapi(FfmpegVaapiBackend),
    VulkanH264(vulkan_h264::AsyncVulkanH264Backend),
    BgraRAW(BgraRAWBackend),
    Gstreamer,
}

pub struct VideoBackend {
    pub backend_type: VideoBackendType,
}

pub struct FrameBuffer {
    pub data: Vec<u8>,
}

impl FrameBuffer {
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.data
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ColorOrder {
    Bgra,
    Rgba,
    Nv12,
    Yuv444p,
}

impl Display for ColorOrder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ColorOrder::Bgra => {
                write!(f, "bgra")
            }
            ColorOrder::Rgba => f.write_str("rgba"),
            ColorOrder::Nv12 => f.write_str("nv12"),
            ColorOrder::Yuv444p => f.write_str("yuv444p"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VideoConfig {
    pub filename: String,
    pub framerate: u32,
    pub output_width: u32,
    pub output_height: u32,
    pub color_order: ColorOrder,
    pub bitrate: Option<u64>,
}

pub enum FfmpegPipeEncoder {
    Libx264,
    Libx265,
    HevcNvenc,
    HevcVaapi,
}

impl FfmpegPipeEncoder {
    fn get_encoder_name(&self) -> &'static str {
        match self {
            Self::Libx264 => "libx264",
            Self::Libx265 => "libx265",
            Self::HevcNvenc => "hevc_nvenc",
            Self::HevcVaapi => "hevc_vaapi",
        }
    }
}

pub struct FfmpegPipeConfig {
    pub ffmpeg_encoder: FfmpegPipeEncoder,
}

pub struct FfmpegPipeBackend {
    child: std::process::Child,
    stdin: Option<std::process::ChildStdin>,
    closed: bool,
    pub frame_size: usize,
}

pub struct FfmpegConfig {
    pub ffmpeg_encoder: FfmpegPipeEncoder,
}

pub struct BgraRAWBackend {
    file: std::fs::File,
    pub frame_size: usize,
}

impl VideoBackend {
    pub fn acquire_buffer(&mut self) -> FrameBuffer {
        match &mut self.backend_type {
            VideoBackendType::VulkanH264(_) => {
                panic!("Vulkan H.264 backend requires GPU frame submission, not CPU buffers")
            }
            VideoBackendType::Vaapi(f) => f.acquire_buffer(),
            VideoBackendType::FfmpegPipe(f) => FrameBuffer {
                data: vec![0u8; f.frame_size],
            },
            VideoBackendType::Ffmpeg(f) => FrameBuffer {
                data: vec![0u8; f.frame_size],
            },
            VideoBackendType::BgraRAW(f) => FrameBuffer {
                data: vec![0u8; f.frame_size],
            },
            VideoBackendType::Gstreamer => unimplemented!(),
        }
    }

    pub fn submit_frame(&mut self, buf: FrameBuffer) {
        match &mut self.backend_type {
            VideoBackendType::VulkanH264(_) => {
                panic!("Vulkan H.264 backend requires GPU frame submission, not CPU buffers")
            }
            VideoBackendType::Vaapi(f) => f.submit_frame(buf),
            VideoBackendType::FfmpegPipe(f) => {
                use std::io::Write;
                if let Some(stdin) = f.stdin.as_mut() {
                    stdin.write_all(&buf.data).unwrap();
                }
            }
            VideoBackendType::Ffmpeg(f) => f.write_frame(&buf.data),
            VideoBackendType::BgraRAW(f) => {
                use std::io::Write;
                f.file.write_all(&buf.data).unwrap();
            }
            _ => {}
        }
    }

    pub fn close(&mut self) -> io::Result<()> {
        match &mut self.backend_type {
            VideoBackendType::FfmpegPipe(f) => f.close(),
            VideoBackendType::Ffmpeg(f) => f.finish(),
            VideoBackendType::Vaapi(f) => f.finish(),
            VideoBackendType::VulkanH264(f) => f.finish(),
            _ => Ok(()),
        }
    }
}
struct FfmpegPipeOutputOptionBuilder {
    high_quality: bool,
    encoder: FfmpegPipeEncoder,
    color_order: ColorOrder,
}

impl FfmpegPipeOutputOptionBuilder {
    fn build_option(&self, args: &mut Vec<String>) {
        args.push("-an".to_string());
        args.extend([
            "-vcodec".to_string(),
            self.encoder.get_encoder_name().to_string(),
        ]);

        self.specify_hwaccel_device_option(args);
        self.specify_quality_option(args);
    }
    fn specify_hwaccel_device_option(&self, args: &mut Vec<String>) {
        if matches!(self.encoder, FfmpegPipeEncoder::HevcVaapi) {
            args.extend([
                "-vaapi_device".to_string(),
                "/dev/dri/renderD128".to_string(),
                "-vf".to_string(),
                "format=nv12,hwupload".to_string(),
            ]);
        }
    }

    fn specify_quality_option(&self, args: &mut Vec<String>) {
        let mut quality_options = match self.encoder {
            FfmpegPipeEncoder::HevcVaapi => {
                if self.high_quality {
                    vec!["-compression_level", "11"] // I can't use level value 1 and 29, and i don't know why.
                } else {
                    vec!["-compression_level", "0"]
                }
            }
            FfmpegPipeEncoder::HevcNvenc => {
                if self.high_quality {
                    vec!["-preset", "p7"]
                } else {
                    vec!["-preset", "p1"]
                }
            }
            _ => {
                if self.high_quality {
                    vec!["-preset", "veryslow"]
                } else {
                    vec!["-preset", "ultrafast"]
                }
            }
        };
        // vaapi only support "vaapi" pix_fmt
        if !matches!(self.encoder, FfmpegPipeEncoder::HevcVaapi) {
            if matches!(self.color_order, ColorOrder::Yuv444p) {
                quality_options.extend(["-pix_fmt", "yuv444p"]);
            } else {
                quality_options.extend(["-pix_fmt", "yuv420p"]);
            }
        }
        args.extend(quality_options.iter().map(|x| x.to_string()))
    }
}

impl FfmpegPipeBackend {
    pub fn new(
        video_config: &VideoConfig,
        encoder_config: FfmpegPipeEncoder,
        high_profile: bool,
    ) -> Self {
        let encoder_name = encoder_config.get_encoder_name();

        let mut args = vec![
            "-y".to_string(),
            "-f".to_string(),
            "rawvideo".to_string(),
            "-pix_fmt".to_string(),
            format!("{}", video_config.color_order).to_string(),
            "-s".to_string(),
            format!(
                "{}x{}",
                video_config.output_width, video_config.output_height
            ),
            "-r".to_string(),
            format!("{}", video_config.framerate),
            "-i".to_string(),
            "-".to_string(),
        ];
        let encoder_option_builder = FfmpegPipeOutputOptionBuilder {
            high_quality: high_profile,
            encoder: encoder_config,
            color_order: video_config.color_order,
        };

        encoder_option_builder.build_option(&mut args);

        args.push(video_config.filename.to_string());

        let mut c = std::process::Command::new("ffmpeg")
            .args(args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("failed to spawn child process");
        let stdin = c.stdin.take().expect("failed to open stdin");
        Self {
            child: c,
            stdin: Some(stdin),
            closed: false,
            frame_size: match video_config.color_order {
                ColorOrder::Nv12 => {
                    let y_size = video_config.output_width * video_config.output_height;
                    let uv_size = video_config.output_width * video_config.output_height / 2;
                    (y_size + uv_size) as usize
                }
                ColorOrder::Yuv444p => {
                    (video_config.output_width * video_config.output_height * 3) as usize
                }
                _ => (video_config.output_width * video_config.output_height * 4) as usize,
            },
        }
    }

    pub fn close(&mut self) -> io::Result<()> {
        if self.closed {
            return Ok(());
        }
        self.stdin.take();
        match self.child.wait() {
            Ok(status) if status.success() => {}
            Ok(status) => {
                return Err(io::Error::other(format!("ffmpeg exited with {status}")));
            }
            Err(e) => return Err(e),
        }
        self.closed = true;
        Ok(())
    }
}

impl Drop for FfmpegPipeBackend {
    fn drop(&mut self) {
        if !self.closed {
            let _ = self.close();
        }
    }
}

impl BgraRAWBackend {
    pub fn new(video_config: &VideoConfig) -> Self {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .open(&format!("{}", video_config.filename))
            .unwrap();
        Self {
            file,
            frame_size: (video_config.output_width * video_config.output_height * 4) as usize,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::video_backend::vaapi::FfmpegVaapiBackend;

    #[test]
    fn vaapi_backend_is_available_through_video_backend_type() {
        let config = VideoConfig {
            filename: "/tmp/gmanim-vaapi-api-test.mp4".to_owned(),
            framerate: 30,
            output_width: 128,
            output_height: 128,
            color_order: ColorOrder::Rgba,
            bitrate: None,
        };

        let mut backend = FfmpegVaapiBackend::try_new(&config, 3).unwrap();

        // Test zero-copy path: acquire → fill → submit
        let mut buf = backend.acquire_buffer();
        buf.as_mut_slice().fill(128);
        backend.submit_frame(buf);

        // Test compatibility path: write_frame(&[u8])
        let frame = vec![64u8; (config.output_width * config.output_height * 4) as usize];
        backend.write_frame(&frame);

        backend.finish().unwrap();
    }
}
