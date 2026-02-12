use std::collections::VecDeque;
use std::fmt::Display;
use std::sync::mpsc::{self, channel, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use ffmpeg_next::format::pixel;

const BLOCK_SIZE: usize = 240;
pub enum VideoBackendType {
    FfmpegPipe(FfmpegPipeBackend),
    Ffmpeg(FfmpegBackend),
    BgraRAW(BgraRAWBackend),
    Gstreamer,
}

pub struct VideoBackend {
    pub backend_type: VideoBackendType,
}

#[derive(Debug, Clone, Copy)]
pub enum ColorOrder {
    Bgra,
    Rgba,
}

impl Display for ColorOrder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ColorOrder::Bgra => {
                write!(f, "bgra")
            }
            ColorOrder::Rgba => {
                write!(f, "rgba")
            }
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
    stdin: std::process::ChildStdin,
}

pub struct FfmpegBackend {
}

impl FfmpegBackend {
    fn new(video_config: &VideoConfig) -> Self {
        ffmpeg_next::init().unwrap();

        // init Muxer
        let mut octx = ffmpeg_next::format::output(&video_config.filename).unwrap();
        let global_header = octx
            .format()
            .flags()
            .contains(ffmpeg_next::format::Flags::GLOBAL_HEADER);
        // config video stream
        let v_codec = ffmpeg_next::encoder::find(ffmpeg_next::codec::Id::MPEG4).unwrap();
        let mut v_stream = octx.add_stream(v_codec).unwrap();
        let mut v_enc =
            ffmpeg_next::codec::context::Context::from_parameters(v_stream.parameters())
                .unwrap()
                .encoder()
                .video()
                .unwrap();
        v_enc.set_width(video_config.output_width);
        v_enc.set_height(video_config.output_height);
        v_enc.set_format(pixel::Pixel::RGBAF32LE);
        v_enc.set_time_base((1 as i32, video_config.framerate as i32));
        if global_header {
            v_enc.set_flags(ffmpeg_next::codec::Flags::GLOBAL_HEADER);
        }
        let mut v_enc = v_enc.open().unwrap();
        v_stream.set_parameters(&v_enc);
        octx.write_header().unwrap();
        
        Self {
        }
    }
}

pub struct FfmpegConfig {
    pub ffmpeg_encoder: FfmpegPipeEncoder,
}

pub struct BgraRAWBackend {
    file: std::fs::File,
}

pub enum FrameMessage {
    Frame,
    End,
}

pub enum FrameDoneMessage {
    Ok,
    Err,
}

#[derive(Clone, Copy)]
pub enum VideoBackendState {
    Running,
    Sleeping,
}

impl VideoBackend {
    pub fn write_frame(&mut self, frame_data: &[u8]) {
        match &mut self.backend_type {
            VideoBackendType::FfmpegPipe(f) => {
                use std::io::Write;
                f.stdin.write_all(frame_data);
            }
            VideoBackendType::BgraRAW(f) => {
                use std::io::Write;
                f.file.write_all(frame_data);
            }
            _ => {}
        }
    }
    pub fn write_frame_background(
        &mut self,
        rx: Receiver<FrameMessage>,
        state: Arc<Mutex<VideoBackendState>>,
        queue: Arc<Mutex<VecDeque<Vec<u8>>>>,
    ) {
        loop {
            let now = std::time::Instant::now();
            let data;
            {
                let mut queue_guard = queue.lock().unwrap();
                data = queue_guard.pop_front();
            }
            if data.is_none() {
                {
                    let mut state_guard = state.lock().unwrap();
                    *state_guard = VideoBackendState::Sleeping;
                }
                println!("sleeping!");
                match rx.recv() {
                    Ok(f) => match f {
                        FrameMessage::Frame => {}
                        FrameMessage::End => {
                            break;
                        }
                    },
                    Err(e) => {
                        //no more frame
                        break;
                    }
                }
            } else {
                self.write_frame(&data.unwrap());
            }
            println!("write takes: {:?}", now.elapsed());
        }
    }
}
struct FfmpegPipeOutputOptionBuilder {
    high_quality: bool,
    encoder: FfmpegPipeEncoder,
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
        match self.encoder {
            FfmpegPipeEncoder::HevcVaapi => {
                args.extend([
                    "-vaapi_device".to_string(),
                    "/dev/dri/renderD128".to_string(),
                    "-vf".to_string(),
                    "format=nv12,hwupload".to_string(),
                ]);
            }
            _ => {}
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
        //vaapi only support "vaapi" pix_fmt
        if !matches!(self.encoder, FfmpegPipeEncoder::HevcVaapi) {
            if self.high_quality {
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
        let mut stdin = c.stdin.take().expect("failed to open stdin");
        Self {
            child: c,
            stdin: stdin,
        }
    }
}

pub struct VideoBackendController {
    // video_backend: Arc<Mutex<VideoBackend>>,
    video_backend: Arc<Mutex<VideoBackend>>,
    background_thread_handler: JoinHandle<()>,
    block_queue: Arc<Mutex<VecDeque<Vec<Vec<u8>>>>>,
    sender: Sender<FrameMessage>,
    block: Option<Vec<Vec<u8>>>,
}

impl VideoBackendController {
    pub fn new(video_backend: VideoBackend) -> Self {
        let video_backend_ref = Arc::new(Mutex::new(video_backend));

        let block_queue = Arc::new(Mutex::new(VecDeque::<Vec<Vec<u8>>>::new()));
        let block_queue_ref = block_queue.clone();

        let video_backend_ref_clone = video_backend_ref.clone();
        let (sender, receiver) = channel::<FrameMessage>();

        let block = Some(Vec::new());

        let handler = thread::spawn(move || {
            let video_backend_ref = video_backend_ref_clone.clone();
            let block_queue = block_queue_ref.clone();
            loop {
                let msg = receiver.recv();
                if msg.is_err() {
                    break;
                }
                let frame_msg = msg.unwrap();
                if matches!(frame_msg, FrameMessage::End) {
                    break;
                }
                let frame_list = match block_queue.lock().unwrap().pop_front() {
                    None => {
                        break;
                    }
                    Some(f) => f,
                };
                let mut video_baackend = video_backend_ref.lock().unwrap();
                for f in frame_list {
                    video_baackend.write_frame(&f);
                }
            }
        });
        Self {
            video_backend: video_backend_ref,
            background_thread_handler: handler,
            block_queue,
            sender,
            block,
        }
    }
    pub fn write_frame(&mut self, frame: Vec<u8>) {
        self.block.as_mut().unwrap().push(frame.to_owned());
        if self.block.as_ref().unwrap().len() == BLOCK_SIZE {
            self.block_queue
                .lock()
                .unwrap()
                .push_back(self.block.replace(Vec::new()).unwrap());
            self.sender.send(FrameMessage::Frame);
        }
    }
    pub fn end(self) {
        self.sender.send(FrameMessage::End);
        self.background_thread_handler.join();
    }
}

impl BgraRAWBackend {
    pub fn new(video_config: &VideoConfig) -> Self {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .open(&format!("{}", video_config.filename))
            .unwrap();
        Self { file }
    }
}
