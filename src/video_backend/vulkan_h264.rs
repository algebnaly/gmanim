use std::collections::{BTreeMap, HashSet};
use std::ffi::CStr;
use std::io::{self, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::{Arc, mpsc};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use ash::khr;
use ash::vk;
use ash::vk::native::{
    StdVideoEncodeH264PictureInfo, StdVideoEncodeH264PictureInfoFlags,
    StdVideoEncodeH264ReferenceInfo, StdVideoEncodeH264ReferenceListsInfo,
    StdVideoEncodeH264SliceHeader, StdVideoEncodeH264SliceHeaderFlags,
    StdVideoH264ChromaFormatIdc_STD_VIDEO_H264_CHROMA_FORMAT_IDC_420,
    StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_4_1, StdVideoH264PictureParameterSet,
    StdVideoH264PictureType_STD_VIDEO_H264_PICTURE_TYPE_IDR,
    StdVideoH264PictureType_STD_VIDEO_H264_PICTURE_TYPE_P,
    StdVideoH264PocType_STD_VIDEO_H264_POC_TYPE_0, StdVideoH264PpsFlags,
    StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_MAIN, StdVideoH264SequenceParameterSet,
    StdVideoH264SequenceParameterSetVui, StdVideoH264SliceType_STD_VIDEO_H264_SLICE_TYPE_I,
    StdVideoH264SliceType_STD_VIDEO_H264_SLICE_TYPE_P, StdVideoH264SpsFlags,
    StdVideoH264SpsVuiFlags,
};

use crate::video_backend::VideoConfig;
use crate::vulkan::context::{TimelineSemaphore, VulkanContext};

pub const VK_KHR_VIDEO_QUEUE_EXTENSION_NAME: &str = "VK_KHR_video_queue";
pub const VK_KHR_VIDEO_ENCODE_QUEUE_EXTENSION_NAME: &str = "VK_KHR_video_encode_queue";
pub const VK_KHR_VIDEO_ENCODE_H264_EXTENSION_NAME: &str = "VK_KHR_video_encode_h264";
pub const REQUIRED_DEVICE_EXTENSIONS: [&str; 3] = [
    VK_KHR_VIDEO_QUEUE_EXTENSION_NAME,
    VK_KHR_VIDEO_ENCODE_QUEUE_EXTENSION_NAME,
    VK_KHR_VIDEO_ENCODE_H264_EXTENSION_NAME,
];

const VK_QUEUE_VIDEO_ENCODE_BIT_KHR_RAW: u32 = 0x0000_0040;
const VK_IMAGE_LAYOUT_VIDEO_ENCODE_SRC_KHR_RAW: i32 = 1_000_299_001;
const VK_IMAGE_LAYOUT_VIDEO_ENCODE_DPB_KHR_RAW: i32 = 1_000_299_002;
const VK_FORMAT_G8_B8R8_2PLANE_420_UNORM_RAW: i32 = 1_000_156_003;
const VK_IMAGE_USAGE_VIDEO_ENCODE_SRC_BIT_KHR_RAW: u32 = 0x0000_4000;
const VK_IMAGE_USAGE_VIDEO_ENCODE_DPB_BIT_KHR_RAW: u32 = 0x0000_8000;
const VK_BUFFER_USAGE_VIDEO_ENCODE_DST_BIT_KHR_RAW: u32 = 0x0000_8000;
const DEFAULT_BITSTREAM_BUFFER_SIZE: u64 = 4 * 1024 * 1024;
const ENCODE_SLOT_COUNT: usize = 8;
const MUX_PACKET_QUEUE_DEPTH: usize = ENCODE_SLOT_COUNT * 2;
const DEFAULT_H264_GOP_SIZE: u32 = 60;
const H264_TARGET_BITRATE: u64 = 8_000_000;
const H264_MAX_BITRATE: u64 = 12_000_000;
const H264_MIN_QP: i32 = 18;
const H264_MAX_QP: i32 = 42;
const STD_VIDEO_H264_NO_REFERENCE_PICTURE: u8 = 0xff;

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct VideoEncodeQueryResult {
    offset: u32,
    bytes: u32,
    status: i32,
}

#[derive(Debug, Clone)]
pub struct VulkanVideoCapabilities {
    pub required_extensions: Vec<&'static str>,
    pub missing_extensions: Vec<&'static str>,
    pub has_video_encode_queue: bool,
    pub h264_profile_query_available: bool,
}

impl VulkanVideoCapabilities {
    pub fn is_usable_for_h264_encode(&self) -> bool {
        self.missing_extensions.is_empty()
            && self.has_video_encode_queue
            && self.h264_profile_query_available
    }

    fn validate(&self) -> io::Result<()> {
        if let Some(missing) = self.missing_extensions.first() {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!("missing Vulkan Video device extension {missing}"),
            ));
        }
        if !self.has_video_encode_queue {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "no Vulkan queue family supports video encode",
            ));
        }
        if !self.h264_profile_query_available {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "Vulkan H.264 profile query is unavailable",
            ));
        }
        Ok(())
    }
}

pub struct VulkanVideoFrame {
    image: vk::Image,
    image_view: vk::ImageView,
    image_layout: vk::ImageLayout,
    format: vk::Format,
    width: u32,
    height: u32,
    device: vk::Device,
    synchronization: Option<VideoFrameSynchronization>,
}

struct VideoFrameSynchronization {
    timeline: Arc<TimelineSemaphore>,
    ready_value: u64,
    release_value: u64,
}

impl VulkanVideoFrame {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        image: vk::Image,
        image_view: vk::ImageView,
        image_layout: vk::ImageLayout,
        format: vk::Format,
        width: u32,
        height: u32,
        device: vk::Device,
        timeline: Arc<TimelineSemaphore>,
        ready_value: u64,
        release_value: u64,
    ) -> Self {
        Self {
            image,
            image_view,
            image_layout,
            format,
            width,
            height,
            device,
            synchronization: Some(VideoFrameSynchronization {
                timeline,
                ready_value,
                release_value,
            }),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum H264RateControlPolicy {
    Vbr,
    Cbr,
    Disabled,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VulkanH264EncoderConfig {
    pub use_p_frames: bool,
    pub gop_size: u32,
    pub rate_control: H264RateControlPolicy,
}

impl Default for VulkanH264EncoderConfig {
    fn default() -> Self {
        Self {
            use_p_frames: true,
            gop_size: DEFAULT_H264_GOP_SIZE,
            rate_control: H264RateControlPolicy::Vbr,
        }
    }
}

impl VulkanH264EncoderConfig {
    fn validate(self) -> io::Result<Self> {
        if self.gop_size == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Vulkan H.264 GOP size must be non-zero",
            ));
        }
        Ok(self)
    }

    fn effective_gop_size(self) -> u32 {
        if self.use_p_frames { self.gop_size } else { 1 }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct VulkanH264Stats {
    pub frames_submitted: u32,
    pub frames_completed: u32,
    pub bitstream_bytes: u64,
    pub completion_span: Duration,
    pub command_record_submit_time: Duration,
    pub packet_readback_time: Duration,
    pub mux_enqueue_time: Duration,
    pub mux_write_time: Duration,
    pub slot_backpressure_waits: u32,
    pub slot_backpressure_time: Duration,
}

impl VulkanH264Stats {
    pub fn average_completion_interval(self) -> Duration {
        self.completion_span
            .checked_div(self.frames_completed.saturating_sub(1))
            .unwrap_or_default()
    }
}

pub struct VulkanH264Backend {
    config: VideoConfig,
    encoder_config: VulkanH264EncoderConfig,
    ctx: Option<Arc<VulkanContext>>,
    capabilities: VulkanVideoCapabilities,
    session: Option<VideoSessionResources>,
    frame_index: u32,
    muxer: Option<AsyncMp4Muxer>,
    pending_packets: BTreeMap<u32, Vec<u8>>,
    next_packet_frame_to_write: u32,
    wrote_header: bool,
    closed: bool,
    stats: VulkanH264Stats,
    first_completion: Option<Instant>,
}

pub struct AsyncVulkanH264Backend {
    capabilities: VulkanVideoCapabilities,
    command_sender: Option<mpsc::SyncSender<AsyncEncoderCommand>>,
    terminal_receiver: mpsc::Receiver<AsyncEncoderTerminal>,
    worker: Option<JoinHandle<()>>,
    closed: bool,
    stats: VulkanH264Stats,
}

enum AsyncEncoderCommand {
    Frame(VulkanVideoFrame),
    Finish,
}

struct AsyncEncoderTerminal {
    result: io::Result<()>,
    stats: VulkanH264Stats,
}

struct AsyncMp4Muxer {
    packet_sender: Option<mpsc::SyncSender<Vec<u8>>>,
    worker: Option<JoinHandle<io::Result<Mp4MuxerStats>>>,
}

#[derive(Clone, Copy, Debug, Default)]
struct Mp4MuxerStats {
    write_time: Duration,
}

struct VideoSessionResources {
    video_queue: khr::video_queue::Device,
    encode_queue: khr::video_encode_queue::Device,
    session: vk::VideoSessionKHR,
    session_parameters: vk::VideoSessionParametersKHR,
    session_memory: Vec<vk::DeviceMemory>,
    sps: StdVideoH264SequenceParameterSet,
    pps: StdVideoH264PictureParameterSet,
    dpb_format: vk::Format,
    max_dpb_slots: u32,
    bitstream_buffer: vk::Buffer,
    bitstream_memory: vk::DeviceMemory,
    bitstream_mapped: *mut u8,
    bitstream_slot_size: u64,
    query_pool: vk::QueryPool,
    dpb_image: vk::Image,
    dpb_memory: vk::DeviceMemory,
    dpb_views: Vec<vk::ImageView>,
    slots: Vec<EncodeSlot>,
    rate_control: Option<H264RateControlSettings>,
}

struct EncodeSlot {
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    query_index: u32,
    frame_index: Option<u32>,
    frame_timeline: Option<Arc<TimelineSemaphore>>,
    busy: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct H264RateControlSettings {
    mode: vk::VideoEncodeRateControlModeFlagsKHR,
    framerate: u32,
    average_bitrate: u64,
    max_bitrate: u64,
    min_qp: i32,
    max_qp: i32,
}

fn default_h264_rate_control_settings(
    mode: vk::VideoEncodeRateControlModeFlagsKHR,
    framerate: u32,
    bitrate: Option<u64>,
) -> H264RateControlSettings {
    let target = bitrate.unwrap_or(H264_TARGET_BITRATE);
    let max_bitrate = if mode == vk::VideoEncodeRateControlModeFlagsKHR::CBR {
        target
    } else {
        H264_MAX_BITRATE.max(target)
    };
    H264RateControlSettings {
        mode,
        framerate,
        average_bitrate: target,
        max_bitrate,
        min_qp: H264_MIN_QP.clamp(0, 51),
        max_qp: H264_MAX_QP.clamp(H264_MIN_QP, 51),
    }
}

fn h264_rate_control_flags(
    encoder_config: VulkanH264EncoderConfig,
) -> vk::VideoEncodeH264RateControlFlagsKHR {
    let mut flags = vk::VideoEncodeH264RateControlFlagsKHR::REGULAR_GOP;
    if encoder_config.use_p_frames {
        flags |= vk::VideoEncodeH264RateControlFlagsKHR::REFERENCE_PATTERN_FLAT;
    }
    flags
}

impl VulkanH264Backend {
    pub fn try_new(ctx: Arc<VulkanContext>, config: &VideoConfig) -> io::Result<Self> {
        Self::try_new_with_encoder_config(ctx, config, VulkanH264EncoderConfig::default())
    }

    pub fn try_new_with_encoder_config(
        ctx: Arc<VulkanContext>,
        config: &VideoConfig,
        encoder_config: VulkanH264EncoderConfig,
    ) -> io::Result<Self> {
        validate_dimensions(config.output_width, config.output_height)?;
        let encoder_config = encoder_config.validate()?;
        let capabilities = detect_capabilities(&ctx)?;
        capabilities.validate()?;
        let session = Some(create_video_session(&ctx, config, encoder_config)?);
        let muxer = AsyncMp4Muxer::try_new(config)?;

        Ok(Self {
            config: config.clone(),
            encoder_config,
            ctx: Some(ctx),
            capabilities,
            session,
            frame_index: 0,
            muxer: Some(muxer),
            pending_packets: BTreeMap::new(),
            next_packet_frame_to_write: 0,
            wrote_header: false,
            closed: false,
            stats: VulkanH264Stats::default(),
            first_completion: None,
        })
    }

    pub fn new(ctx: Arc<VulkanContext>, config: &VideoConfig) -> Self {
        Self::try_new(ctx, config).expect("failed to create Vulkan H.264 backend")
    }

    #[cfg(test)]
    pub fn new_for_test(config: VideoConfig, capabilities: VulkanVideoCapabilities) -> Self {
        Self {
            config,
            encoder_config: VulkanH264EncoderConfig::default(),
            ctx: None,
            capabilities,
            session: None,
            frame_index: 0,
            muxer: None,
            pending_packets: BTreeMap::new(),
            next_packet_frame_to_write: 0,
            wrote_header: false,
            closed: false,
            stats: VulkanH264Stats::default(),
            first_completion: None,
        }
    }

    pub fn capabilities(&self) -> &VulkanVideoCapabilities {
        &self.capabilities
    }

    pub fn stats(&self) -> VulkanH264Stats {
        self.stats
    }

    fn record_completed_packet(&mut self, bytes: usize) {
        let now = Instant::now();
        let first = *self.first_completion.get_or_insert(now);
        self.stats.frames_completed += 1;
        self.stats.bitstream_bytes += bytes as u64;
        self.stats.completion_span = now.duration_since(first);
    }

    pub fn submit_vulkan_frame(&mut self, frame: VulkanVideoFrame) -> io::Result<()> {
        if self.closed {
            return Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "Vulkan H.264 backend is already closed",
            ));
        }
        if self.session.is_none() {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "Vulkan Video H.264 bitstream encoding is not initialized",
            ));
        }
        let ctx = self
            .ctx
            .as_ref()
            .ok_or_else(|| io::Error::other("Vulkan context is not initialized"))?;
        validate_frame(&self.config, &frame, ctx)?;
        if !self.wrote_header {
            write_h264_headers(self)?;
            self.wrote_header = true;
        }
        collect_completed_packets(self, false)?;
        let slot_index = acquire_encode_slot(self)?;
        let submit_start = Instant::now();
        encode_one_frame(self, frame, slot_index)?;
        self.stats.command_record_submit_time += submit_start.elapsed();
        self.frame_index += 1;
        self.stats.frames_submitted += 1;
        Ok(())
    }

    pub fn finish(&mut self) -> io::Result<()> {
        if self.closed {
            return Ok(());
        }
        collect_completed_packets(self, true)?;
        if let Some(mut muxer) = self.muxer.take() {
            let muxer_stats = muxer.finish()?;
            self.stats.mux_write_time = muxer_stats.write_time;
        }
        if let (Some(ctx), Some(mut session)) = (self.ctx.as_ref(), self.session.take()) {
            unsafe {
                ctx.device.device_wait_idle().map_err(io::Error::other)?;
            }
            destroy_video_session_resources(ctx, &mut session);
        }
        self.closed = true;
        Ok(())
    }
}

impl Drop for VulkanH264Backend {
    fn drop(&mut self) {
        let _ = self.finish();
    }
}

impl AsyncVulkanH264Backend {
    pub fn try_new(ctx: Arc<VulkanContext>, config: &VideoConfig) -> io::Result<Self> {
        Self::try_new_with_encoder_config(ctx, config, VulkanH264EncoderConfig::default())
    }

    pub fn try_new_with_encoder_config(
        ctx: Arc<VulkanContext>,
        config: &VideoConfig,
        encoder_config: VulkanH264EncoderConfig,
    ) -> io::Result<Self> {
        let encoder_config = encoder_config.validate()?;
        let config = config.clone();
        let (command_sender, command_receiver) = mpsc::sync_channel(ENCODE_SLOT_COUNT);
        let (initialization_sender, initialization_receiver) = mpsc::sync_channel(1);
        let (terminal_sender, terminal_receiver) = mpsc::sync_channel(1);
        let worker = thread::Builder::new()
            .name("gmanim-vulkan-h264".to_owned())
            .spawn(move || {
                let mut backend = match VulkanH264Backend::try_new_with_encoder_config(
                    ctx,
                    &config,
                    encoder_config,
                ) {
                    Ok(backend) => backend,
                    Err(error) => {
                        let _ = initialization_sender.send(Err(error));
                        return;
                    }
                };
                if initialization_sender
                    .send(Ok(backend.capabilities().clone()))
                    .is_err()
                {
                    return;
                }

                let encode_result = loop {
                    match command_receiver.recv() {
                        Ok(AsyncEncoderCommand::Frame(frame)) => {
                            if let Err(error) = backend.submit_vulkan_frame(frame) {
                                break Err(error);
                            }
                        }
                        Ok(AsyncEncoderCommand::Finish) | Err(_) => break Ok(()),
                    }
                };
                let finish_result = backend.finish();
                let _ = terminal_sender.send(AsyncEncoderTerminal {
                    result: encode_result.and(finish_result),
                    stats: backend.stats(),
                });
            })
            .map_err(io::Error::other)?;

        let capabilities = match initialization_receiver.recv() {
            Ok(Ok(capabilities)) => capabilities,
            Ok(Err(error)) => {
                let _ = worker.join();
                return Err(error);
            }
            Err(error) => {
                let _ = worker.join();
                return Err(io::Error::other(format!(
                    "Vulkan H.264 worker initialization failed: {error}"
                )));
            }
        };

        Ok(Self {
            capabilities,
            command_sender: Some(command_sender),
            terminal_receiver,
            worker: Some(worker),
            closed: false,
            stats: VulkanH264Stats::default(),
        })
    }

    pub fn capabilities(&self) -> &VulkanVideoCapabilities {
        &self.capabilities
    }

    pub fn stats(&self) -> VulkanH264Stats {
        self.stats
    }

    pub fn submit_vulkan_frame(&mut self, frame: VulkanVideoFrame) -> io::Result<()> {
        if self.closed {
            return Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "asynchronous Vulkan H.264 backend is already closed",
            ));
        }
        self.command_sender
            .as_ref()
            .ok_or_else(|| io::Error::other("Vulkan H.264 worker is unavailable"))?
            .send(AsyncEncoderCommand::Frame(frame))
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "Vulkan H.264 worker stopped"))
    }

    pub fn finish(&mut self) -> io::Result<()> {
        if self.closed {
            return Ok(());
        }
        if let Some(sender) = self.command_sender.take() {
            let _ = sender.send(AsyncEncoderCommand::Finish);
        }
        if let Some(worker) = self.worker.take() {
            worker
                .join()
                .map_err(|_| io::Error::other("Vulkan H.264 worker panicked"))?;
        }
        self.closed = true;
        let terminal = self.terminal_receiver.recv().map_err(|error| {
            io::Error::other(format!("Vulkan H.264 worker returned no result: {error}"))
        })?;
        self.stats = terminal.stats;
        terminal.result
    }
}

impl Drop for AsyncVulkanH264Backend {
    fn drop(&mut self) {
        let _ = self.finish();
    }
}

impl AsyncMp4Muxer {
    fn try_new(config: &VideoConfig) -> io::Result<Self> {
        let (mut child, mut stdin) = spawn_mp4_muxer_process(config)?;
        let (packet_sender, packet_receiver) =
            mpsc::sync_channel::<Vec<u8>>(MUX_PACKET_QUEUE_DEPTH);
        let worker = thread::Builder::new()
            .name("gmanim-ffmpeg-mux".to_owned())
            .spawn(move || {
                let mut stats = Mp4MuxerStats::default();
                let write_result = (|| {
                    while let Ok(packet) = packet_receiver.recv() {
                        let write_start = Instant::now();
                        stdin.write_all(&packet)?;
                        stats.write_time += write_start.elapsed();
                    }
                    Ok(())
                })();
                drop(stdin);
                let wait_result = child.wait().and_then(|status| {
                    if status.success() {
                        Ok(())
                    } else {
                        Err(io::Error::other(format!(
                            "ffmpeg muxer exited with {status}"
                        )))
                    }
                });
                write_result.and(wait_result)?;
                Ok(stats)
            })
            .map_err(io::Error::other)?;
        Ok(Self {
            packet_sender: Some(packet_sender),
            worker: Some(worker),
        })
    }

    fn submit(&self, packet: Vec<u8>) -> io::Result<Duration> {
        let enqueue_start = Instant::now();
        self.packet_sender
            .as_ref()
            .ok_or_else(|| io::Error::new(io::ErrorKind::BrokenPipe, "FFmpeg muxer is closed"))?
            .send(packet)
            .map_err(|_| {
                io::Error::new(io::ErrorKind::BrokenPipe, "FFmpeg muxer worker stopped")
            })?;
        Ok(enqueue_start.elapsed())
    }

    fn finish(&mut self) -> io::Result<Mp4MuxerStats> {
        self.packet_sender.take();
        self.worker
            .take()
            .map(|worker| {
                worker
                    .join()
                    .map_err(|_| io::Error::other("FFmpeg muxer worker panicked"))?
            })
            .unwrap_or_else(|| Ok(Mp4MuxerStats::default()))
    }
}

impl Drop for AsyncMp4Muxer {
    fn drop(&mut self) {
        let _ = self.finish();
    }
}

fn spawn_mp4_muxer_process(config: &VideoConfig) -> io::Result<(Child, ChildStdin)> {
    if config.output_color_profile != crate::OutputColorProfile::Bt709Sdr {
        return Err(io::Error::other(
            "Vulkan H.264 currently supports only 8-bit BT.709 SDR output",
        ));
    }
    let mut child = Command::new("ffmpeg")
        .args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            &config.framerate.to_string(),
            "-f",
            "h264",
            "-i",
            "-",
            "-c:v",
            "copy",
            "-color_range",
            "tv",
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-movflags",
            "+faststart",
            &config.filename,
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;
    let stdin = child
        .stdin
        .take()
        .ok_or_else(|| io::Error::other("failed to open ffmpeg stdin"))?;
    Ok((child, stdin))
}

pub fn missing_required_extensions<'a>(
    available_extensions: impl IntoIterator<Item = &'a str>,
) -> Vec<&'static str> {
    let available: HashSet<&str> = available_extensions.into_iter().collect();
    REQUIRED_DEVICE_EXTENSIONS
        .iter()
        .copied()
        .filter(|required| !available.contains(required))
        .collect()
}

pub fn validate_dimensions(width: u32, height: u32) -> io::Result<()> {
    if width == 0 || height == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Vulkan H.264 video dimensions must be non-zero",
        ));
    }
    if !width.is_multiple_of(2) || !height.is_multiple_of(2) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Vulkan H.264 NV12 video dimensions must be even",
        ));
    }
    Ok(())
}

fn detect_capabilities(ctx: &VulkanContext) -> io::Result<VulkanVideoCapabilities> {
    let extension_names = unsafe {
        ctx.instance
            .enumerate_device_extension_properties(ctx.physical_device)
            .map_err(io::Error::other)?
    };
    let extension_name_refs: Vec<String> = extension_names
        .iter()
        .map(|extension| unsafe {
            CStr::from_ptr(extension.extension_name.as_ptr())
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    let missing_extensions =
        missing_required_extensions(extension_name_refs.iter().map(String::as_str));

    let queue_families = unsafe {
        ctx.instance
            .get_physical_device_queue_family_properties(ctx.physical_device)
    };
    let has_video_encode_queue = queue_families
        .iter()
        .any(|props| props.queue_flags.as_raw() & VK_QUEUE_VIDEO_ENCODE_BIT_KHR_RAW != 0);
    let h264_profile_query_available = query_h264_encode_profile_support(ctx).unwrap_or(false);

    Ok(VulkanVideoCapabilities {
        required_extensions: REQUIRED_DEVICE_EXTENSIONS.to_vec(),
        missing_extensions,
        has_video_encode_queue,
        h264_profile_query_available,
    })
}

fn query_h264_encode_profile_support(ctx: &VulkanContext) -> io::Result<bool> {
    let mut encode_usage = vk::VideoEncodeUsageInfoKHR::default()
        .video_usage_hints(vk::VideoEncodeUsageFlagsKHR::RECORDING)
        .video_content_hints(vk::VideoEncodeContentFlagsKHR::RENDERED)
        .tuning_mode(vk::VideoEncodeTuningModeKHR::HIGH_QUALITY);
    let mut h264_profile = vk::VideoEncodeH264ProfileInfoKHR::default()
        .std_profile_idc(StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_MAIN);
    let profile = vk::VideoProfileInfoKHR::default()
        .video_codec_operation(vk::VideoCodecOperationFlagsKHR::ENCODE_H264)
        .chroma_subsampling(vk::VideoChromaSubsamplingFlagsKHR::TYPE_420)
        .luma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
        .chroma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
        .push_next(&mut encode_usage)
        .push_next(&mut h264_profile);
    let mut h264_caps = vk::VideoEncodeH264CapabilitiesKHR::default();
    let mut encode_caps = vk::VideoEncodeCapabilitiesKHR {
        p_next: (&mut h264_caps as *mut vk::VideoEncodeH264CapabilitiesKHR).cast(),
        ..Default::default()
    };
    let mut video_caps = vk::VideoCapabilitiesKHR::default().push_next(&mut encode_caps);
    let video_queue = khr::video_queue::Instance::new(&ctx.entry, &ctx.instance);

    let result = unsafe {
        (video_queue.fp().get_physical_device_video_capabilities_khr)(
            ctx.physical_device,
            &profile,
            &mut video_caps,
        )
    };

    match result {
        vk::Result::SUCCESS => Ok(true),
        vk::Result::ERROR_VIDEO_PROFILE_CODEC_NOT_SUPPORTED_KHR
        | vk::Result::ERROR_VIDEO_PROFILE_FORMAT_NOT_SUPPORTED_KHR
        | vk::Result::ERROR_VIDEO_PROFILE_OPERATION_NOT_SUPPORTED_KHR => Ok(false),
        err => Err(io::Error::other(format!(
            "vkGetPhysicalDeviceVideoCapabilitiesKHR failed: {err:?}"
        ))),
    }
}

fn create_video_session(
    ctx: &Arc<VulkanContext>,
    config: &VideoConfig,
    encoder_config: VulkanH264EncoderConfig,
) -> io::Result<VideoSessionResources> {
    let video_queue = khr::video_queue::Device::new(&ctx.instance, &ctx.device);
    let encode_queue = khr::video_encode_queue::Device::new(&ctx.instance, &ctx.device);
    let queue_family_index = ctx.video_encode_queue_family_index.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::Unsupported,
            "no Vulkan queue family supports video encode",
        )
    })?;
    let mut h264_profile = vk::VideoEncodeH264ProfileInfoKHR::default()
        .std_profile_idc(StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_MAIN);
    let profile = vk::VideoProfileInfoKHR::default()
        .video_codec_operation(vk::VideoCodecOperationFlagsKHR::ENCODE_H264)
        .chroma_subsampling(vk::VideoChromaSubsamplingFlagsKHR::TYPE_420)
        .luma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
        .chroma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
        .push_next(&mut h264_profile);

    let mut h264_caps = vk::VideoEncodeH264CapabilitiesKHR::default();
    let mut encode_caps = vk::VideoEncodeCapabilitiesKHR {
        p_next: (&mut h264_caps as *mut vk::VideoEncodeH264CapabilitiesKHR).cast(),
        ..Default::default()
    };
    let mut video_caps = vk::VideoCapabilitiesKHR::default().push_next(&mut encode_caps);
    let video_instance = khr::video_queue::Instance::new(&ctx.entry, &ctx.instance);
    let result = unsafe {
        (video_instance
            .fp()
            .get_physical_device_video_capabilities_khr)(
            ctx.physical_device,
            &profile,
            &mut video_caps,
        )
    };
    if result != vk::Result::SUCCESS {
        return Err(io::Error::other(format!(
            "vkGetPhysicalDeviceVideoCapabilitiesKHR failed: {result:?}"
        )));
    }

    let picture_format = choose_video_format(
        ctx,
        &video_instance,
        profile,
        vk::ImageUsageFlags::from_raw(VK_IMAGE_USAGE_VIDEO_ENCODE_SRC_BIT_KHR_RAW),
        |format| format.as_raw() == VK_FORMAT_G8_B8R8_2PLANE_420_UNORM_RAW,
    )?;
    let dpb_format = choose_video_format(
        ctx,
        &video_instance,
        profile,
        vk::ImageUsageFlags::from_raw(VK_IMAGE_USAGE_VIDEO_ENCODE_DPB_BIT_KHR_RAW),
        |_| true,
    )?;

    let max_dpb_slots = video_caps.max_dpb_slots.clamp(1, 2);
    let mut h264_session_create_info = vk::VideoEncodeH264SessionCreateInfoKHR::default();
    let create_info = vk::VideoSessionCreateInfoKHR::default()
        .queue_family_index(queue_family_index)
        .video_profile(&profile)
        .picture_format(picture_format)
        .max_coded_extent(vk::Extent2D {
            width: config.output_width,
            height: config.output_height,
        })
        .reference_picture_format(dpb_format)
        .max_dpb_slots(max_dpb_slots)
        .max_active_reference_pictures(1)
        .std_header_version(&video_caps.std_header_version)
        .push_next(&mut h264_session_create_info);

    let mut session = vk::VideoSessionKHR::null();
    let result = unsafe {
        (video_queue.fp().create_video_session_khr)(
            ctx.device.handle(),
            &create_info,
            std::ptr::null(),
            &mut session,
        )
    };
    if result != vk::Result::SUCCESS {
        return Err(io::Error::other(format!(
            "vkCreateVideoSessionKHR failed: {result:?}"
        )));
    }

    let rate_control = make_h264_rate_control_settings(
        &encode_caps,
        config.framerate,
        config.bitrate,
        encoder_config.rate_control,
    )?;
    let mut resources = VideoSessionResources {
        video_queue,
        encode_queue,
        session,
        session_parameters: vk::VideoSessionParametersKHR::null(),
        session_memory: Vec::new(),
        sps: make_h264_sps(config.output_width, config.output_height, None),
        pps: make_h264_pps(),
        dpb_format,
        max_dpb_slots,
        bitstream_buffer: vk::Buffer::null(),
        bitstream_memory: vk::DeviceMemory::null(),
        bitstream_mapped: std::ptr::null_mut(),
        bitstream_slot_size: DEFAULT_BITSTREAM_BUFFER_SIZE,
        query_pool: vk::QueryPool::null(),
        dpb_image: vk::Image::null(),
        dpb_memory: vk::DeviceMemory::null(),
        dpb_views: Vec::new(),
        slots: Vec::new(),
        rate_control,
    };

    let result = bind_video_session_memory(ctx, &mut resources)
        .and_then(|_| create_video_encode_resources(ctx, &mut resources, config, &profile))
        .and_then(|_| create_video_session_parameters(ctx, &mut resources, config))
        .and_then(|_| configure_h264_rate_control(ctx, &mut resources, encoder_config));
    if let Err(err) = result {
        destroy_video_session_resources(ctx, &mut resources);
        return Err(err);
    }

    Ok(resources)
}

fn make_h264_rate_control_settings(
    encode_caps: &vk::VideoEncodeCapabilitiesKHR,
    framerate: u32,
    bitrate: Option<u64>,
    policy: H264RateControlPolicy,
) -> io::Result<Option<H264RateControlSettings>> {
    let mode = match policy {
        H264RateControlPolicy::Disabled => return Ok(None),
        H264RateControlPolicy::Vbr => vk::VideoEncodeRateControlModeFlagsKHR::VBR,
        H264RateControlPolicy::Cbr => vk::VideoEncodeRateControlModeFlagsKHR::CBR,
    };
    if !encode_caps.rate_control_modes.contains(mode) {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            format!(
                "requested Vulkan H.264 rate-control mode {policy:?} is unsupported; driver reports {:?}",
                encode_caps.rate_control_modes
            ),
        ));
    }
    Ok(Some(default_h264_rate_control_settings(
        mode, framerate, bitrate,
    )))
}

fn choose_video_format(
    ctx: &VulkanContext,
    video_instance: &khr::video_queue::Instance,
    profile: vk::VideoProfileInfoKHR,
    image_usage: vk::ImageUsageFlags,
    accepts: impl Fn(vk::Format) -> bool,
) -> io::Result<vk::Format> {
    let profiles = [profile];
    let mut profile_list = vk::VideoProfileListInfoKHR::default().profiles(&profiles);
    let format_info = vk::PhysicalDeviceVideoFormatInfoKHR::default()
        .image_usage(image_usage)
        .push_next(&mut profile_list);
    let mut count = 0;
    let result = unsafe {
        (video_instance
            .fp()
            .get_physical_device_video_format_properties_khr)(
            ctx.physical_device,
            &format_info,
            &mut count,
            std::ptr::null_mut(),
        )
    };
    if result != vk::Result::SUCCESS {
        return Err(io::Error::other(format!(
            "vkGetPhysicalDeviceVideoFormatPropertiesKHR(count) failed: {result:?}"
        )));
    }
    let mut formats = vec![vk::VideoFormatPropertiesKHR::default(); count as usize];
    let result = unsafe {
        (video_instance
            .fp()
            .get_physical_device_video_format_properties_khr)(
            ctx.physical_device,
            &format_info,
            &mut count,
            formats.as_mut_ptr(),
        )
    };
    if result != vk::Result::SUCCESS {
        return Err(io::Error::other(format!(
            "vkGetPhysicalDeviceVideoFormatPropertiesKHR failed: {result:?}"
        )));
    }
    formats
        .into_iter()
        .map(|props| props.format)
        .find(|format| accepts(*format))
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Unsupported,
                "no compatible Vulkan Video image format was reported",
            )
        })
}

fn bind_video_session_memory(
    ctx: &VulkanContext,
    resources: &mut VideoSessionResources,
) -> io::Result<()> {
    let mut count = 0;
    let result = unsafe {
        (resources
            .video_queue
            .fp()
            .get_video_session_memory_requirements_khr)(
            ctx.device.handle(),
            resources.session,
            &mut count,
            std::ptr::null_mut(),
        )
    };
    if result != vk::Result::SUCCESS {
        return Err(io::Error::other(format!(
            "vkGetVideoSessionMemoryRequirementsKHR(count) failed: {result:?}"
        )));
    }
    let mut requirements = vec![vk::VideoSessionMemoryRequirementsKHR::default(); count as usize];
    if count > 0 {
        let result = unsafe {
            (resources
                .video_queue
                .fp()
                .get_video_session_memory_requirements_khr)(
                ctx.device.handle(),
                resources.session,
                &mut count,
                requirements.as_mut_ptr(),
            )
        };
        if result != vk::Result::SUCCESS {
            return Err(io::Error::other(format!(
                "vkGetVideoSessionMemoryRequirementsKHR failed: {result:?}"
            )));
        }
    }

    let mut bind_infos = Vec::with_capacity(requirements.len());
    for requirement in requirements {
        let memory_type_index = find_memory_type_index(
            ctx,
            requirement.memory_requirements.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let allocate_info = vk::MemoryAllocateInfo {
            s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
            allocation_size: requirement.memory_requirements.size,
            memory_type_index,
            ..Default::default()
        };
        let memory = unsafe {
            ctx.device
                .allocate_memory(&allocate_info, None)
                .map_err(io::Error::other)?
        };
        bind_infos.push(
            vk::BindVideoSessionMemoryInfoKHR::default()
                .memory_bind_index(requirement.memory_bind_index)
                .memory(memory)
                .memory_offset(0)
                .memory_size(requirement.memory_requirements.size),
        );
        resources.session_memory.push(memory);
    }
    if !bind_infos.is_empty() {
        let result = unsafe {
            (resources.video_queue.fp().bind_video_session_memory_khr)(
                ctx.device.handle(),
                resources.session,
                bind_infos.len() as u32,
                bind_infos.as_ptr(),
            )
        };
        if result != vk::Result::SUCCESS {
            return Err(io::Error::other(format!(
                "vkBindVideoSessionMemoryKHR failed: {result:?}"
            )));
        }
    }
    Ok(())
}

fn create_video_encode_resources(
    ctx: &VulkanContext,
    resources: &mut VideoSessionResources,
    config: &VideoConfig,
    profile: &vk::VideoProfileInfoKHR,
) -> io::Result<()> {
    let profiles = [*profile];
    let mut profile_list = vk::VideoProfileListInfoKHR::default().profiles(&profiles);
    let buffer_info = vk::BufferCreateInfo {
        s_type: vk::StructureType::BUFFER_CREATE_INFO,
        p_next: (&mut profile_list as *mut vk::VideoProfileListInfoKHR).cast(),
        size: DEFAULT_BITSTREAM_BUFFER_SIZE * ENCODE_SLOT_COUNT as u64,
        usage: vk::BufferUsageFlags::from_raw(VK_BUFFER_USAGE_VIDEO_ENCODE_DST_BIT_KHR_RAW),
        sharing_mode: vk::SharingMode::EXCLUSIVE,
        ..Default::default()
    };
    resources.bitstream_buffer = unsafe {
        ctx.device
            .create_buffer(&buffer_info, None)
            .map_err(io::Error::other)?
    };
    let buffer_requirements = unsafe {
        ctx.device
            .get_buffer_memory_requirements(resources.bitstream_buffer)
    };
    let buffer_memory_type_index = find_memory_type_index(
        ctx,
        buffer_requirements.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    let buffer_allocate_info = vk::MemoryAllocateInfo {
        s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
        allocation_size: buffer_requirements.size,
        memory_type_index: buffer_memory_type_index,
        ..Default::default()
    };
    resources.bitstream_memory = unsafe {
        ctx.device
            .allocate_memory(&buffer_allocate_info, None)
            .map_err(io::Error::other)?
    };
    unsafe {
        ctx.device
            .bind_buffer_memory(resources.bitstream_buffer, resources.bitstream_memory, 0)
            .map_err(io::Error::other)?;
        resources.bitstream_mapped = ctx
            .device
            .map_memory(
                resources.bitstream_memory,
                0,
                DEFAULT_BITSTREAM_BUFFER_SIZE * ENCODE_SLOT_COUNT as u64,
                vk::MemoryMapFlags::empty(),
            )
            .map_err(io::Error::other)? as *mut u8;
    }

    let dpb_layers = resources.max_dpb_slots.max(1);
    let mut dpb_profile_list = vk::VideoProfileListInfoKHR::default().profiles(&profiles);
    let dpb_image_info = vk::ImageCreateInfo {
        s_type: vk::StructureType::IMAGE_CREATE_INFO,
        p_next: (&mut dpb_profile_list as *mut vk::VideoProfileListInfoKHR).cast(),
        image_type: vk::ImageType::TYPE_2D,
        format: resources.dpb_format,
        extent: vk::Extent3D {
            width: config.output_width,
            height: config.output_height,
            depth: 1,
        },
        mip_levels: 1,
        array_layers: dpb_layers,
        samples: vk::SampleCountFlags::TYPE_1,
        tiling: vk::ImageTiling::OPTIMAL,
        usage: vk::ImageUsageFlags::from_raw(VK_IMAGE_USAGE_VIDEO_ENCODE_DPB_BIT_KHR_RAW),
        sharing_mode: vk::SharingMode::EXCLUSIVE,
        initial_layout: vk::ImageLayout::UNDEFINED,
        ..Default::default()
    };
    resources.dpb_image = unsafe {
        ctx.device
            .create_image(&dpb_image_info, None)
            .map_err(io::Error::other)?
    };
    let dpb_requirements = unsafe {
        ctx.device
            .get_image_memory_requirements(resources.dpb_image)
    };
    let dpb_memory_type_index = find_memory_type_index(
        ctx,
        dpb_requirements.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    let dpb_allocate_info = vk::MemoryAllocateInfo {
        s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
        allocation_size: dpb_requirements.size,
        memory_type_index: dpb_memory_type_index,
        ..Default::default()
    };
    resources.dpb_memory = unsafe {
        ctx.device
            .allocate_memory(&dpb_allocate_info, None)
            .map_err(io::Error::other)?
    };
    unsafe {
        ctx.device
            .bind_image_memory(resources.dpb_image, resources.dpb_memory, 0)
            .map_err(io::Error::other)?;
    }
    for layer in 0..dpb_layers {
        let view_info = vk::ImageViewCreateInfo {
            s_type: vk::StructureType::IMAGE_VIEW_CREATE_INFO,
            image: resources.dpb_image,
            view_type: vk::ImageViewType::TYPE_2D_ARRAY,
            format: resources.dpb_format,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: layer,
                layer_count: 1,
            },
            ..Default::default()
        };
        resources.dpb_views.push(unsafe {
            ctx.device
                .create_image_view(&view_info, None)
                .map_err(io::Error::other)?
        });
    }

    let mut feedback_info = vk::QueryPoolVideoEncodeFeedbackCreateInfoKHR::default()
        .encode_feedback_flags(
            vk::VideoEncodeFeedbackFlagsKHR::BITSTREAM_BUFFER_OFFSET
                | vk::VideoEncodeFeedbackFlagsKHR::BITSTREAM_BYTES_WRITTEN,
        );
    feedback_info.p_next = (profile as *const vk::VideoProfileInfoKHR).cast();
    let query_pool_info = vk::QueryPoolCreateInfo {
        s_type: vk::StructureType::QUERY_POOL_CREATE_INFO,
        p_next: (&mut feedback_info as *mut vk::QueryPoolVideoEncodeFeedbackCreateInfoKHR).cast(),
        query_type: vk::QueryType::VIDEO_ENCODE_FEEDBACK_KHR,
        query_count: ENCODE_SLOT_COUNT as u32,
        ..Default::default()
    };
    resources.query_pool = unsafe {
        ctx.device
            .create_query_pool(&query_pool_info, None)
            .map_err(io::Error::other)?
    };

    let queue_family_index = ctx.video_encode_queue_family_index.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::Unsupported,
            "no Vulkan queue family supports video encode",
        )
    })?;
    for query_index in 0..ENCODE_SLOT_COUNT as u32 {
        let command_pool_info = vk::CommandPoolCreateInfo {
            s_type: vk::StructureType::COMMAND_POOL_CREATE_INFO,
            queue_family_index,
            flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            ..Default::default()
        };
        let command_pool = unsafe {
            ctx.device
                .create_command_pool(&command_pool_info, None)
                .map_err(io::Error::other)?
        };
        let command_buffer_info = vk::CommandBufferAllocateInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
            command_pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: 1,
            ..Default::default()
        };
        let command_buffer = unsafe {
            ctx.device
                .allocate_command_buffers(&command_buffer_info)
                .map_err(io::Error::other)?[0]
        };
        let fence_info = vk::FenceCreateInfo {
            s_type: vk::StructureType::FENCE_CREATE_INFO,
            flags: vk::FenceCreateFlags::SIGNALED,
            ..Default::default()
        };
        let fence = unsafe {
            ctx.device
                .create_fence(&fence_info, None)
                .map_err(io::Error::other)?
        };
        resources.slots.push(EncodeSlot {
            command_pool,
            command_buffer,
            fence,
            query_index,
            frame_index: None,
            frame_timeline: None,
            busy: false,
        });
    }
    Ok(())
}

fn create_video_session_parameters(
    ctx: &VulkanContext,
    resources: &mut VideoSessionResources,
    config: &VideoConfig,
) -> io::Result<()> {
    let vui = make_h264_vui(config.framerate);
    resources.sps = make_h264_sps(config.output_width, config.output_height, Some(&vui));
    resources.pps = make_h264_pps();
    let sps = [resources.sps];
    let pps = [resources.pps];
    let add_info = vk::VideoEncodeH264SessionParametersAddInfoKHR::default()
        .std_sp_ss(&sps)
        .std_pp_ss(&pps);
    let mut h264_create_info = vk::VideoEncodeH264SessionParametersCreateInfoKHR::default()
        .max_std_sps_count(1)
        .max_std_pps_count(1)
        .parameters_add_info(&add_info);
    let create_info = vk::VideoSessionParametersCreateInfoKHR::default()
        .video_session(resources.session)
        .push_next(&mut h264_create_info);
    let result = unsafe {
        (resources
            .video_queue
            .fp()
            .create_video_session_parameters_khr)(
            ctx.device.handle(),
            &create_info,
            std::ptr::null(),
            &mut resources.session_parameters,
        )
    };
    if result != vk::Result::SUCCESS {
        return Err(io::Error::other(format!(
            "vkCreateVideoSessionParametersKHR failed: {result:?}"
        )));
    }
    Ok(())
}

fn configure_h264_rate_control(
    ctx: &VulkanContext,
    resources: &mut VideoSessionResources,
    encoder_config: VulkanH264EncoderConfig,
) -> io::Result<()> {
    let settings = match resources.rate_control {
        Some(settings) => settings,
        None => return Ok(()),
    };
    if resources.slots.is_empty() {
        return Ok(());
    }

    let slot = &resources.slots[0];
    let mut h264_layer = vk::VideoEncodeH264RateControlLayerInfoKHR::default()
        .use_min_qp(true)
        .min_qp(
            vk::VideoEncodeH264QpKHR::default()
                .qp_i(settings.min_qp)
                .qp_p(settings.min_qp)
                .qp_b(settings.min_qp),
        )
        .use_max_qp(true)
        .max_qp(
            vk::VideoEncodeH264QpKHR::default()
                .qp_i(settings.max_qp)
                .qp_p(settings.max_qp)
                .qp_b(settings.max_qp),
        );
    let layer = vk::VideoEncodeRateControlLayerInfoKHR::default()
        .frame_rate_numerator(settings.framerate)
        .frame_rate_denominator(1)
        .average_bitrate(settings.average_bitrate)
        .max_bitrate(settings.max_bitrate)
        .push_next(&mut h264_layer);
    let layers = [layer];
    let gop_size = encoder_config.effective_gop_size();
    let mut h264_info = vk::VideoEncodeH264RateControlInfoKHR::default()
        .flags(h264_rate_control_flags(encoder_config))
        .gop_frame_count(gop_size)
        .idr_period(gop_size)
        .consecutive_b_frame_count(0)
        .temporal_layer_count(1);
    let mut rc_info = vk::VideoEncodeRateControlInfoKHR::default()
        .rate_control_mode(settings.mode)
        .layers(&layers)
        .initial_virtual_buffer_size_in_ms(100)
        .virtual_buffer_size_in_ms(200);
    rc_info.p_next = (&mut h264_info as *mut vk::VideoEncodeH264RateControlInfoKHR).cast();

    unsafe {
        ctx.device
            .wait_for_fences(std::slice::from_ref(&slot.fence), true, u64::MAX)
            .map_err(io::Error::other)?;
        ctx.device
            .reset_fences(std::slice::from_ref(&slot.fence))
            .map_err(io::Error::other)?;
        ctx.device
            .reset_command_pool(slot.command_pool, vk::CommandPoolResetFlags::empty())
            .map_err(io::Error::other)?;
        let begin_info = vk::CommandBufferBeginInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            ..Default::default()
        };
        ctx.device
            .begin_command_buffer(slot.command_buffer, &begin_info)
            .map_err(io::Error::other)?;
        let begin_coding = vk::VideoBeginCodingInfoKHR::default()
            .video_session(resources.session)
            .video_session_parameters(resources.session_parameters);
        (resources.video_queue.fp().cmd_begin_video_coding_khr)(slot.command_buffer, &begin_coding);
        let control = vk::VideoCodingControlInfoKHR::default()
            .flags(
                vk::VideoCodingControlFlagsKHR::RESET
                    | vk::VideoCodingControlFlagsKHR::ENCODE_RATE_CONTROL,
            )
            .push_next(&mut rc_info);
        (resources.video_queue.fp().cmd_control_video_coding_khr)(slot.command_buffer, &control);
        let end_coding = vk::VideoEndCodingInfoKHR::default();
        (resources.video_queue.fp().cmd_end_video_coding_khr)(slot.command_buffer, &end_coding);
        ctx.device
            .end_command_buffer(slot.command_buffer)
            .map_err(io::Error::other)?;
        let submit_info = vk::SubmitInfo {
            s_type: vk::StructureType::SUBMIT_INFO,
            command_buffer_count: 1,
            p_command_buffers: &slot.command_buffer,
            ..Default::default()
        };
        let queue = ctx.video_encode_queue.unwrap_or(ctx.queue);
        ctx.device
            .queue_submit(queue, std::slice::from_ref(&submit_info), slot.fence)
            .map_err(io::Error::other)?;
        ctx.device
            .wait_for_fences(std::slice::from_ref(&slot.fence), true, u64::MAX)
            .map_err(io::Error::other)?;
    }

    Ok(())
}

fn write_h264_headers(backend: &mut VulkanH264Backend) -> io::Result<()> {
    let resources = backend
        .session
        .as_ref()
        .ok_or_else(|| io::Error::other("Vulkan video session is not initialized"))?;
    let ctx = backend
        .ctx
        .as_ref()
        .ok_or_else(|| io::Error::other("Vulkan context is not initialized"))?;
    let mut h264_get = vk::VideoEncodeH264SessionParametersGetInfoKHR::default()
        .write_std_sps(true)
        .write_std_pps(true)
        .std_sps_id(0)
        .std_pps_id(0);
    let get_info = vk::VideoEncodeSessionParametersGetInfoKHR::default()
        .video_session_parameters(resources.session_parameters)
        .push_next(&mut h264_get);
    let mut len = 0usize;
    let result = unsafe {
        (resources
            .encode_queue
            .fp()
            .get_encoded_video_session_parameters_khr)(
            ctx.device.handle(),
            &get_info,
            std::ptr::null_mut(),
            &mut len,
            std::ptr::null_mut(),
        )
    };
    if result != vk::Result::SUCCESS {
        return Err(io::Error::other(format!(
            "vkGetEncodedVideoSessionParametersKHR(count) failed: {result:?}"
        )));
    }
    let mut data = vec![0u8; len];
    let mut h264_feedback = vk::VideoEncodeH264SessionParametersFeedbackInfoKHR::default();
    let mut feedback =
        vk::VideoEncodeSessionParametersFeedbackInfoKHR::default().push_next(&mut h264_feedback);
    let result = unsafe {
        (resources
            .encode_queue
            .fp()
            .get_encoded_video_session_parameters_khr)(
            ctx.device.handle(),
            &get_info,
            &mut feedback,
            &mut len,
            data.as_mut_ptr().cast(),
        )
    };
    if result != vk::Result::SUCCESS {
        return Err(io::Error::other(format!(
            "vkGetEncodedVideoSessionParametersKHR failed: {result:?}"
        )));
    }
    data.truncate(len);
    if let Some(muxer) = backend.muxer.as_ref() {
        backend.stats.mux_enqueue_time += muxer.submit(data)?;
    }
    Ok(())
}

fn encode_one_frame(
    backend: &mut VulkanH264Backend,
    mut frame: VulkanVideoFrame,
    slot_index: usize,
) -> io::Result<()> {
    let ctx = backend
        .ctx
        .as_ref()
        .ok_or_else(|| io::Error::other("Vulkan context is not initialized"))?;
    let resources = backend
        .session
        .as_mut()
        .ok_or_else(|| io::Error::other("Vulkan video session is not initialized"))?;
    let slot = resources
        .slots
        .get_mut(slot_index)
        .ok_or_else(|| io::Error::other("invalid Vulkan Video encode slot"))?;
    let frame_num = backend.frame_index;
    let encoder_config = backend.encoder_config;
    let gop_size = encoder_config.effective_gop_size();
    let gop_frame_num = frame_num % gop_size;
    let is_idr =
        !encoder_config.use_p_frames || gop_frame_num == 0 || resources.dpb_views.len() < 2;
    let dpb_slot_index = if is_idr {
        0
    } else {
        (gop_frame_num as usize) & 1
    };
    let ref_slot_index = ((gop_frame_num.saturating_sub(1) as usize) & 1)
        .min(resources.dpb_views.len().saturating_sub(1));
    let dpb_ref_info = make_h264_reference_info(gop_frame_num, is_idr, &resources.sps);
    let mut dpb_slot_h264 =
        vk::VideoEncodeH264DpbSlotInfoKHR::default().std_reference_info(&dpb_ref_info);
    let ref_ref_info = make_h264_reference_info(
        gop_frame_num.saturating_sub(1),
        gop_frame_num.saturating_sub(1) == 0,
        &resources.sps,
    );
    let mut ref_slot_h264 =
        vk::VideoEncodeH264DpbSlotInfoKHR::default().std_reference_info(&ref_ref_info);
    let extent = vk::Extent2D {
        width: frame.width,
        height: frame.height,
    };
    let dpb_picture = vk::VideoPictureResourceInfoKHR::default()
        .image_view_binding(resources.dpb_views[dpb_slot_index])
        .coded_extent(extent);
    let ref_picture = vk::VideoPictureResourceInfoKHR::default()
        .image_view_binding(resources.dpb_views[ref_slot_index])
        .coded_extent(extent);
    let mut setup_slot = vk::VideoReferenceSlotInfoKHR::default()
        .slot_index(-1)
        .picture_resource(&dpb_picture)
        .push_next(&mut dpb_slot_h264);
    let reference_slot = vk::VideoReferenceSlotInfoKHR::default()
        .slot_index(ref_slot_index as i32)
        .picture_resource(&ref_picture)
        .push_next(&mut ref_slot_h264);
    let begin_slots = if is_idr {
        vec![setup_slot]
    } else {
        vec![setup_slot, reference_slot]
    };
    let encode_reference_slots = if is_idr {
        Vec::new()
    } else {
        vec![reference_slot]
    };
    let mut frame_info = H264FrameInfo::new(
        gop_frame_num,
        &resources.sps,
        &resources.pps,
        is_idr,
        ref_slot_index as u8,
    );
    let src_picture = vk::VideoPictureResourceInfoKHR::default()
        .image_view_binding(frame.image_view)
        .coded_extent(extent);

    unsafe {
        ctx.device
            .reset_fences(std::slice::from_ref(&slot.fence))
            .map_err(io::Error::other)?;
        ctx.device
            .reset_command_pool(slot.command_pool, vk::CommandPoolResetFlags::empty())
            .map_err(io::Error::other)?;
        let begin_info = vk::CommandBufferBeginInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            ..Default::default()
        };
        ctx.device
            .begin_command_buffer(slot.command_buffer, &begin_info)
            .map_err(io::Error::other)?;
        ctx.device.cmd_reset_query_pool(
            slot.command_buffer,
            resources.query_pool,
            slot.query_index,
            1,
        );

        if let Some(settings) = resources.rate_control {
            let mut h264_layer = vk::VideoEncodeH264RateControlLayerInfoKHR::default()
                .use_min_qp(true)
                .min_qp(
                    vk::VideoEncodeH264QpKHR::default()
                        .qp_i(settings.min_qp)
                        .qp_p(settings.min_qp)
                        .qp_b(settings.min_qp),
                )
                .use_max_qp(true)
                .max_qp(
                    vk::VideoEncodeH264QpKHR::default()
                        .qp_i(settings.max_qp)
                        .qp_p(settings.max_qp)
                        .qp_b(settings.max_qp),
                );
            let layer = vk::VideoEncodeRateControlLayerInfoKHR::default()
                .frame_rate_numerator(settings.framerate)
                .frame_rate_denominator(1)
                .average_bitrate(settings.average_bitrate)
                .max_bitrate(settings.max_bitrate)
                .push_next(&mut h264_layer);
            let layers = [layer];
            let mut h264_info = vk::VideoEncodeH264RateControlInfoKHR::default()
                .flags(h264_rate_control_flags(encoder_config))
                .gop_frame_count(gop_size)
                .idr_period(gop_size)
                .consecutive_b_frame_count(0)
                .temporal_layer_count(1);
            let mut rc_info = vk::VideoEncodeRateControlInfoKHR::default()
                .rate_control_mode(settings.mode)
                .layers(&layers)
                .initial_virtual_buffer_size_in_ms(100)
                .virtual_buffer_size_in_ms(200);
            rc_info.p_next = (&mut h264_info as *mut vk::VideoEncodeH264RateControlInfoKHR).cast();
            let begin_coding = vk::VideoBeginCodingInfoKHR::default()
                .video_session(resources.session)
                .video_session_parameters(resources.session_parameters)
                .reference_slots(&begin_slots)
                .push_next(&mut rc_info);
            (resources.video_queue.fp().cmd_begin_video_coding_khr)(
                slot.command_buffer,
                &begin_coding,
            );
        } else {
            let begin_coding = vk::VideoBeginCodingInfoKHR::default()
                .video_session(resources.session)
                .video_session_parameters(resources.session_parameters)
                .reference_slots(&begin_slots);
            (resources.video_queue.fp().cmd_begin_video_coding_khr)(
                slot.command_buffer,
                &begin_coding,
            );
        }

        let src_to_encode = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::NONE)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::VIDEO_ENCODE_KHR)
            .dst_access_mask(vk::AccessFlags2::VIDEO_ENCODE_READ_KHR)
            .old_layout(frame.image_layout)
            .new_layout(vk::ImageLayout::VIDEO_ENCODE_SRC_KHR)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(frame.image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });
        let dpb_to_encode = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::NONE)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::VIDEO_ENCODE_KHR)
            .dst_access_mask(
                vk::AccessFlags2::VIDEO_ENCODE_READ_KHR | vk::AccessFlags2::VIDEO_ENCODE_WRITE_KHR,
            )
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::from_raw(
                VK_IMAGE_LAYOUT_VIDEO_ENCODE_DPB_KHR_RAW,
            ))
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(resources.dpb_image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: dpb_slot_index as u32,
                layer_count: 1,
            });
        let encode_barriers = [src_to_encode, dpb_to_encode];
        let encode_dependency =
            vk::DependencyInfo::default().image_memory_barriers(&encode_barriers);
        ctx.device
            .cmd_pipeline_barrier2(slot.command_buffer, &encode_dependency);

        setup_slot.slot_index = dpb_slot_index as i32;
        let bitstream_offset = resources.bitstream_slot_size * slot_index as u64;
        let encode_info = vk::VideoEncodeInfoKHR::default()
            .dst_buffer(resources.bitstream_buffer)
            .dst_buffer_offset(bitstream_offset)
            .dst_buffer_range(resources.bitstream_slot_size)
            .src_picture_resource(src_picture)
            .reference_slots(&encode_reference_slots)
            .setup_reference_slot(&setup_slot)
            .push_next(&mut frame_info.picture_info);
        ctx.device.cmd_begin_query(
            slot.command_buffer,
            resources.query_pool,
            slot.query_index,
            vk::QueryControlFlags::empty(),
        );
        (resources.encode_queue.fp().cmd_encode_video_khr)(slot.command_buffer, &encode_info);
        ctx.device
            .cmd_end_query(slot.command_buffer, resources.query_pool, slot.query_index);
        let end_coding = vk::VideoEndCodingInfoKHR::default();
        (resources.video_queue.fp().cmd_end_video_coding_khr)(slot.command_buffer, &end_coding);
        let src_to_general = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::VIDEO_ENCODE_KHR)
            .src_access_mask(vk::AccessFlags2::VIDEO_ENCODE_READ_KHR)
            .dst_stage_mask(vk::PipelineStageFlags2::NONE)
            .dst_access_mask(vk::AccessFlags2::NONE)
            .old_layout(vk::ImageLayout::VIDEO_ENCODE_SRC_KHR)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(frame.image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });
        let release_dependency = vk::DependencyInfo::default()
            .image_memory_barriers(std::slice::from_ref(&src_to_general));
        ctx.device
            .cmd_pipeline_barrier2(slot.command_buffer, &release_dependency);
        ctx.device
            .end_command_buffer(slot.command_buffer)
            .map_err(io::Error::other)?;
        let synchronization = frame
            .synchronization
            .as_ref()
            .ok_or_else(|| io::Error::other("Vulkan video frame has no synchronization lease"))?;
        let wait_info = vk::SemaphoreSubmitInfo::default()
            .semaphore(synchronization.timeline.handle())
            .value(synchronization.ready_value)
            .stage_mask(vk::PipelineStageFlags2::VIDEO_ENCODE_KHR);
        let signal_info = vk::SemaphoreSubmitInfo::default()
            .semaphore(synchronization.timeline.handle())
            .value(synchronization.release_value)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS);
        let command_buffer_info =
            vk::CommandBufferSubmitInfo::default().command_buffer(slot.command_buffer);
        let submit_info = vk::SubmitInfo2::default()
            .wait_semaphore_infos(std::slice::from_ref(&wait_info))
            .command_buffer_infos(std::slice::from_ref(&command_buffer_info))
            .signal_semaphore_infos(std::slice::from_ref(&signal_info));
        let queue = ctx.video_encode_queue.unwrap_or(ctx.queue);
        ctx.device
            .queue_submit2(queue, std::slice::from_ref(&submit_info), slot.fence)
            .map_err(io::Error::other)?;
    }

    let synchronization = frame
        .synchronization
        .take()
        .ok_or_else(|| io::Error::other("Vulkan video frame synchronization was lost"))?;
    slot.busy = true;
    slot.frame_index = Some(frame_num);
    slot.frame_timeline = Some(synchronization.timeline);
    Ok(())
}

fn collect_completed_packets(backend: &mut VulkanH264Backend, wait_all: bool) -> io::Result<()> {
    let ctx = backend
        .ctx
        .as_ref()
        .ok_or_else(|| io::Error::other("Vulkan context is not initialized"))?;
    let mut completed_packets = Vec::new();
    {
        let resources = match backend.session.as_mut() {
            Some(resources) => resources,
            None => return Ok(()),
        };

        for slot_index in 0..resources.slots.len() {
            if !resources.slots[slot_index].busy {
                continue;
            }

            let fence = resources.slots[slot_index].fence;
            let ready = if wait_all {
                unsafe {
                    ctx.device
                        .wait_for_fences(std::slice::from_ref(&fence), true, u64::MAX)
                        .map_err(io::Error::other)?;
                }
                true
            } else {
                match unsafe { ctx.device.get_fence_status(fence) } {
                    Ok(true) => true,
                    Ok(false) => false,
                    Err(err) => return Err(io::Error::other(err)),
                }
            };

            if !ready {
                continue;
            }

            let frame_index = resources.slots[slot_index].frame_index.ok_or_else(|| {
                io::Error::other("completed Vulkan Video slot has no frame index")
            })?;
            let readback_start = Instant::now();
            let packet = read_completed_slot_packet(ctx, resources, slot_index)?;
            completed_packets.push((frame_index, packet, readback_start.elapsed()));
            resources.slots[slot_index].busy = false;
            resources.slots[slot_index].frame_index = None;
            resources.slots[slot_index].frame_timeline = None;
        }
    }

    if !completed_packets.is_empty() {
        for (frame_index, packet, readback_time) in completed_packets {
            backend.record_completed_packet(packet.len());
            backend.stats.packet_readback_time += readback_time;
            backend.pending_packets.insert(frame_index, packet);
        }
        flush_ordered_packets(backend)?;
    }
    Ok(())
}

fn acquire_encode_slot(backend: &mut VulkanH264Backend) -> io::Result<usize> {
    if let Some(resources) = backend.session.as_ref()
        && let Some(index) = resources.slots.iter().position(|slot| !slot.busy)
    {
        return Ok(index);
    }

    let ctx = backend
        .ctx
        .as_ref()
        .ok_or_else(|| io::Error::other("Vulkan context is not initialized"))?;
    let (slot_index, frame_index, packet, wait_time, readback_time) = {
        let resources = backend
            .session
            .as_mut()
            .ok_or_else(|| io::Error::other("Vulkan video session is not initialized"))?;
        let slot_index = resources
            .slots
            .iter()
            .position(|slot| slot.busy)
            .ok_or_else(|| io::Error::other("Vulkan Video has no encode slots"))?;
        let fence = resources.slots[slot_index].fence;
        let wait_start = Instant::now();
        unsafe {
            ctx.device
                .wait_for_fences(std::slice::from_ref(&fence), true, u64::MAX)
                .map_err(io::Error::other)?;
        }
        let wait_time = wait_start.elapsed();
        let frame_index = resources.slots[slot_index]
            .frame_index
            .ok_or_else(|| io::Error::other("completed Vulkan Video slot has no frame index"))?;
        let readback_start = Instant::now();
        let packet = read_completed_slot_packet(ctx, resources, slot_index)?;
        let readback_time = readback_start.elapsed();
        resources.slots[slot_index].busy = false;
        resources.slots[slot_index].frame_index = None;
        resources.slots[slot_index].frame_timeline = None;
        (slot_index, frame_index, packet, wait_time, readback_time)
    };

    backend.stats.slot_backpressure_waits += 1;
    backend.stats.slot_backpressure_time += wait_time;
    backend.stats.packet_readback_time += readback_time;
    backend.record_completed_packet(packet.len());
    backend.pending_packets.insert(frame_index, packet);
    flush_ordered_packets(backend)?;
    Ok(slot_index)
}

fn flush_ordered_packets(backend: &mut VulkanH264Backend) -> io::Result<()> {
    let packets = drain_ordered_packets(
        &mut backend.pending_packets,
        &mut backend.next_packet_frame_to_write,
    );
    if let Some(muxer) = backend.muxer.as_ref() {
        for packet in packets {
            backend.stats.mux_enqueue_time += muxer.submit(packet)?;
        }
    }
    Ok(())
}

fn drain_ordered_packets(
    pending: &mut BTreeMap<u32, Vec<u8>>,
    next_frame: &mut u32,
) -> Vec<Vec<u8>> {
    let mut packets = Vec::new();
    while let Some(packet) = pending.remove(next_frame) {
        packets.push(packet);
        *next_frame += 1;
    }
    packets
}

fn read_completed_slot_packet(
    ctx: &VulkanContext,
    resources: &mut VideoSessionResources,
    slot_index: usize,
) -> io::Result<Vec<u8>> {
    let slot = resources
        .slots
        .get(slot_index)
        .ok_or_else(|| io::Error::other("invalid Vulkan Video encode slot"))?;
    let mut result = VideoEncodeQueryResult::default();
    unsafe {
        ctx.device
            .get_query_pool_results(
                resources.query_pool,
                slot.query_index,
                std::slice::from_mut(&mut result),
                vk::QueryResultFlags::WITH_STATUS_KHR,
            )
            .map_err(io::Error::other)?;
    }
    if result.bytes == 0 {
        return Err(io::Error::other(
            "Vulkan Video encoder produced an empty packet",
        ));
    }
    let slot_offset = resources.bitstream_slot_size as usize * slot_index;
    let offset = slot_offset + result.offset as usize;
    let end = offset + result.bytes as usize;
    let slot_end = slot_offset + resources.bitstream_slot_size as usize;
    if end > slot_end {
        return Err(io::Error::other(
            "Vulkan Video query returned out-of-range bitstream packet",
        ));
    }
    let packet = unsafe {
        std::slice::from_raw_parts(
            resources.bitstream_mapped.add(offset),
            result.bytes as usize,
        )
    };
    Ok(packet.to_vec())
}

fn destroy_video_session_resources(ctx: &VulkanContext, resources: &mut VideoSessionResources) {
    unsafe {
        for slot in resources.slots.drain(..) {
            if slot.fence != vk::Fence::null() {
                ctx.device.destroy_fence(slot.fence, None);
            }
            if slot.command_pool != vk::CommandPool::null() {
                ctx.device.destroy_command_pool(slot.command_pool, None);
            }
        }
        for view in resources.dpb_views.drain(..) {
            ctx.device.destroy_image_view(view, None);
        }
        if resources.dpb_image != vk::Image::null() {
            ctx.device.destroy_image(resources.dpb_image, None);
        }
        if resources.dpb_memory != vk::DeviceMemory::null() {
            ctx.device.free_memory(resources.dpb_memory, None);
        }
        if resources.query_pool != vk::QueryPool::null() {
            ctx.device.destroy_query_pool(resources.query_pool, None);
        }
        if !resources.bitstream_mapped.is_null() {
            ctx.device.unmap_memory(resources.bitstream_memory);
        }
        if resources.bitstream_buffer != vk::Buffer::null() {
            ctx.device.destroy_buffer(resources.bitstream_buffer, None);
        }
        if resources.bitstream_memory != vk::DeviceMemory::null() {
            ctx.device.free_memory(resources.bitstream_memory, None);
        }
        if resources.session_parameters != vk::VideoSessionParametersKHR::null() {
            (resources
                .video_queue
                .fp()
                .destroy_video_session_parameters_khr)(
                ctx.device.handle(),
                resources.session_parameters,
                std::ptr::null(),
            );
        }
        if resources.session != vk::VideoSessionKHR::null() {
            (resources.video_queue.fp().destroy_video_session_khr)(
                ctx.device.handle(),
                resources.session,
                std::ptr::null(),
            );
        }
        for memory in resources.session_memory.drain(..) {
            ctx.device.free_memory(memory, None);
        }
    }
}

fn find_memory_type_index(
    ctx: &VulkanContext,
    memory_type_bits: u32,
    required: vk::MemoryPropertyFlags,
) -> io::Result<u32> {
    let memory_properties = unsafe {
        ctx.instance
            .get_physical_device_memory_properties(ctx.physical_device)
    };
    for index in 0..memory_properties.memory_type_count {
        let supported = memory_type_bits & (1 << index) != 0;
        let flags = memory_properties.memory_types[index as usize].property_flags;
        if supported && flags.contains(required) {
            return Ok(index);
        }
    }
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "no compatible Vulkan memory type was found",
    ))
}

fn validate_frame(
    config: &VideoConfig,
    frame: &VulkanVideoFrame,
    context: &VulkanContext,
) -> io::Result<()> {
    if frame.device != context.device.handle() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Vulkan video frame belongs to a different device",
        ));
    }
    if frame.synchronization.is_none() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Vulkan video frame has no synchronization lease",
        ));
    }
    validate_dimensions(frame.width, frame.height)?;
    if frame.width != config.output_width || frame.height != config.output_height {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "Vulkan frame dimensions {}x{} do not match video config {}x{}",
                frame.width, frame.height, config.output_width, config.output_height
            ),
        ));
    }
    if frame.image == vk::Image::null() || frame.image_view == vk::ImageView::null() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Vulkan video frame image and image view must not be null",
        ));
    }
    if frame.format.as_raw() != VK_FORMAT_G8_B8R8_2PLANE_420_UNORM_RAW {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Vulkan H.264 backend currently accepts only NV12 images",
        ));
    }
    if frame.image_layout != vk::ImageLayout::GENERAL
        && frame.image_layout.as_raw() != VK_IMAGE_LAYOUT_VIDEO_ENCODE_SRC_KHR_RAW
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Vulkan video frame image layout must be VIDEO_ENCODE_SRC_KHR or GENERAL",
        ));
    }
    Ok(())
}

fn make_h264_vui(framerate: u32) -> StdVideoH264SequenceParameterSetVui {
    let mut flags: StdVideoH264SpsVuiFlags = unsafe { std::mem::zeroed() };
    flags.set_timing_info_present_flag(1);
    flags.set_fixed_frame_rate_flag(1);
    StdVideoH264SequenceParameterSetVui {
        flags,
        aspect_ratio_idc: 0,
        sar_width: 0,
        sar_height: 0,
        video_format: 0,
        colour_primaries: 0,
        transfer_characteristics: 0,
        matrix_coefficients: 0,
        num_units_in_tick: 1,
        time_scale: framerate * 2,
        max_num_reorder_frames: 0,
        max_dec_frame_buffering: 0,
        chroma_sample_loc_type_top_field: 0,
        chroma_sample_loc_type_bottom_field: 0,
        reserved1: 0,
        pHrdParameters: std::ptr::null(),
    }
}

fn make_h264_sps(
    width: u32,
    height: u32,
    vui: Option<&StdVideoH264SequenceParameterSetVui>,
) -> StdVideoH264SequenceParameterSet {
    let aligned_width = align_up(width, 16);
    let aligned_height = align_up(height, 16);
    let mut flags: StdVideoH264SpsFlags = unsafe { std::mem::zeroed() };
    flags.set_direct_8x8_inference_flag(1);
    flags.set_frame_mbs_only_flag(1);
    flags.set_vui_parameters_present_flag(u32::from(vui.is_some()));
    let mut frame_crop_right_offset = aligned_width - width;
    let mut frame_crop_bottom_offset = aligned_height - height;
    if frame_crop_right_offset != 0 || frame_crop_bottom_offset != 0 {
        flags.set_frame_cropping_flag(1);
        frame_crop_right_offset >>= 1;
        frame_crop_bottom_offset >>= 1;
    }
    StdVideoH264SequenceParameterSet {
        flags,
        profile_idc: StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_MAIN,
        level_idc: StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_4_1,
        chroma_format_idc: StdVideoH264ChromaFormatIdc_STD_VIDEO_H264_CHROMA_FORMAT_IDC_420,
        seq_parameter_set_id: 0,
        bit_depth_luma_minus8: 0,
        bit_depth_chroma_minus8: 0,
        log2_max_frame_num_minus4: 0,
        pic_order_cnt_type: StdVideoH264PocType_STD_VIDEO_H264_POC_TYPE_0,
        offset_for_non_ref_pic: 0,
        offset_for_top_to_bottom_field: 0,
        log2_max_pic_order_cnt_lsb_minus4: 4,
        num_ref_frames_in_pic_order_cnt_cycle: 0,
        max_num_ref_frames: 16,
        reserved1: 0,
        pic_width_in_mbs_minus1: aligned_width / 16 - 1,
        pic_height_in_map_units_minus1: aligned_height / 16 - 1,
        frame_crop_left_offset: 0,
        frame_crop_right_offset,
        frame_crop_top_offset: 0,
        frame_crop_bottom_offset,
        reserved2: 0,
        pOffsetForRefFrame: std::ptr::null(),
        pScalingLists: std::ptr::null(),
        pSequenceParameterSetVui: vui.map_or(std::ptr::null(), |vui| vui as *const _),
    }
}

fn make_h264_pps() -> StdVideoH264PictureParameterSet {
    let mut flags: StdVideoH264PpsFlags = unsafe { std::mem::zeroed() };
    flags.set_deblocking_filter_control_present_flag(1);
    flags.set_entropy_coding_mode_flag(1);
    StdVideoH264PictureParameterSet {
        flags,
        seq_parameter_set_id: 0,
        pic_parameter_set_id: 0,
        num_ref_idx_l0_default_active_minus1: 0,
        num_ref_idx_l1_default_active_minus1: 0,
        weighted_bipred_idc: 0,
        pic_init_qp_minus26: 0,
        pic_init_qs_minus26: 0,
        chroma_qp_index_offset: 0,
        second_chroma_qp_index_offset: 0,
        pScalingLists: std::ptr::null(),
    }
}

fn align_up(value: u32, alignment: u32) -> u32 {
    (value + alignment - 1) & !(alignment - 1)
}

struct H264FrameInfo {
    _slice_header: Box<StdVideoEncodeH264SliceHeader>,
    _slice_info: Box<vk::VideoEncodeH264NaluSliceInfoKHR<'static>>,
    _ref_lists: Box<StdVideoEncodeH264ReferenceListsInfo>,
    _picture: Box<StdVideoEncodeH264PictureInfo>,
    picture_info: vk::VideoEncodeH264PictureInfoKHR<'static>,
}

impl H264FrameInfo {
    fn new(
        gop_frame_num: u32,
        sps: &StdVideoH264SequenceParameterSet,
        pps: &StdVideoH264PictureParameterSet,
        is_idr: bool,
        reference_slot_index: u8,
    ) -> Self {
        let max_pic_order_cnt_lsb = 1u32 << (sps.log2_max_pic_order_cnt_lsb_minus4 + 4);
        let mut slice_flags: StdVideoEncodeH264SliceHeaderFlags = unsafe { std::mem::zeroed() };
        slice_flags.set_direct_spatial_mv_pred_flag(1);
        let slice_header = Box::new(StdVideoEncodeH264SliceHeader {
            flags: slice_flags,
            first_mb_in_slice: 0,
            slice_type: if is_idr {
                StdVideoH264SliceType_STD_VIDEO_H264_SLICE_TYPE_I
            } else {
                StdVideoH264SliceType_STD_VIDEO_H264_SLICE_TYPE_P
            },
            slice_alpha_c0_offset_div2: 0,
            slice_beta_offset_div2: 0,
            slice_qp_delta: 0,
            reserved1: 0,
            cabac_init_idc: 0,
            disable_deblocking_filter_idc: 0,
            pWeightTable: std::ptr::null(),
        });
        let mut slice_info = Box::new(vk::VideoEncodeH264NaluSliceInfoKHR::default());
        slice_info.p_std_slice_header = &*slice_header;

        let mut ref_lists: StdVideoEncodeH264ReferenceListsInfo = unsafe { std::mem::zeroed() };
        ref_lists.RefPicList0 = [STD_VIDEO_H264_NO_REFERENCE_PICTURE; 32];
        ref_lists.RefPicList1 = [STD_VIDEO_H264_NO_REFERENCE_PICTURE; 32];
        ref_lists.num_ref_idx_l0_active_minus1 = 0;
        ref_lists.num_ref_idx_l1_active_minus1 = 0;
        if !is_idr {
            ref_lists.RefPicList0[0] = reference_slot_index;
        }
        let ref_lists = Box::new(ref_lists);

        let mut picture_flags: StdVideoEncodeH264PictureInfoFlags = unsafe { std::mem::zeroed() };
        picture_flags.set_IdrPicFlag(u32::from(is_idr));
        picture_flags.set_is_reference(1);
        picture_flags.set_no_output_of_prior_pics_flag(u32::from(is_idr));
        let mut picture = Box::new(StdVideoEncodeH264PictureInfo {
            flags: picture_flags,
            seq_parameter_set_id: 0,
            pic_parameter_set_id: pps.pic_parameter_set_id,
            idr_pic_id: 0,
            primary_pic_type: if is_idr {
                StdVideoH264PictureType_STD_VIDEO_H264_PICTURE_TYPE_IDR
            } else {
                StdVideoH264PictureType_STD_VIDEO_H264_PICTURE_TYPE_P
            },
            frame_num: h264_frame_num(gop_frame_num, sps),
            PicOrderCnt: ((gop_frame_num * 2) % max_pic_order_cnt_lsb) as i32,
            temporal_id: 0,
            reserved1: [0; 3],
            pRefLists: std::ptr::null(),
        });
        picture.pRefLists = &*ref_lists;
        let picture_info = vk::VideoEncodeH264PictureInfoKHR {
            s_type: vk::StructureType::VIDEO_ENCODE_H264_PICTURE_INFO_KHR,
            nalu_slice_entry_count: 1,
            p_nalu_slice_entries: &*slice_info,
            p_std_picture_info: &*picture,
            ..Default::default()
        };
        Self {
            _slice_header: slice_header,
            _slice_info: slice_info,
            _ref_lists: ref_lists,
            _picture: picture,
            picture_info,
        }
    }
}

fn h264_frame_num(gop_frame_num: u32, sps: &StdVideoH264SequenceParameterSet) -> u32 {
    let max_frame_num = 1u32 << (sps.log2_max_frame_num_minus4 + 4);
    gop_frame_num % max_frame_num
}

fn make_h264_reference_info(
    gop_frame_num: u32,
    is_idr: bool,
    sps: &StdVideoH264SequenceParameterSet,
) -> StdVideoEncodeH264ReferenceInfo {
    let max_pic_order_cnt_lsb = 1u32 << (sps.log2_max_pic_order_cnt_lsb_minus4 + 4);
    StdVideoEncodeH264ReferenceInfo {
        flags: unsafe { std::mem::zeroed() },
        primary_pic_type: if is_idr {
            StdVideoH264PictureType_STD_VIDEO_H264_PICTURE_TYPE_IDR
        } else {
            StdVideoH264PictureType_STD_VIDEO_H264_PICTURE_TYPE_P
        },
        FrameNum: h264_frame_num(gop_frame_num, sps),
        PicOrderCnt: ((gop_frame_num * 2) % max_pic_order_cnt_lsb) as i32,
        long_term_pic_num: 0,
        long_term_frame_idx: 0,
        temporal_id: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::video_backend::ColorOrder;
    use ash::vk::Handle;

    fn config(width: u32, height: u32) -> VideoConfig {
        VideoConfig {
            filename: "out.mp4".to_string(),
            framerate: 30,
            output_width: width,
            output_height: height,
            color_order: ColorOrder::Nv12,
            bitrate: None,
            output_color_profile: Default::default(),
        }
    }

    #[test]
    fn required_extension_names_are_exact() {
        assert_eq!(VK_KHR_VIDEO_QUEUE_EXTENSION_NAME, "VK_KHR_video_queue");
        assert_eq!(
            VK_KHR_VIDEO_ENCODE_QUEUE_EXTENSION_NAME,
            "VK_KHR_video_encode_queue"
        );
        assert_eq!(
            VK_KHR_VIDEO_ENCODE_H264_EXTENSION_NAME,
            "VK_KHR_video_encode_h264"
        );
    }

    #[test]
    fn missing_required_extensions_reports_only_absent_names() {
        let missing = missing_required_extensions([
            "VK_KHR_video_queue",
            "VK_KHR_video_encode_h264",
            "VK_EXT_memory_budget",
        ]);

        assert_eq!(missing, vec!["VK_KHR_video_encode_queue"]);
    }

    #[test]
    fn validation_rejects_odd_dimensions() {
        let err = validate_dimensions(1919, 1080).unwrap_err();
        assert!(err.to_string().contains("even"));
    }

    #[test]
    fn cpu_less_test_backend_rejects_gpu_submission_without_real_session() {
        let caps = VulkanVideoCapabilities {
            required_extensions: REQUIRED_DEVICE_EXTENSIONS.to_vec(),
            missing_extensions: Vec::new(),
            has_video_encode_queue: true,
            h264_profile_query_available: true,
        };
        let mut backend = VulkanH264Backend::new_for_test(config(1920, 1080), caps);
        let frame = VulkanVideoFrame {
            image: vk::Image::from_raw(1),
            image_view: vk::ImageView::from_raw(2),
            image_layout: vk::ImageLayout::GENERAL,
            format: vk::Format::from_raw(VK_FORMAT_G8_B8R8_2PLANE_420_UNORM_RAW),
            width: 1920,
            height: 1080,
            device: vk::Device::null(),
            synchronization: None,
        };

        let err = backend.submit_vulkan_frame(frame).unwrap_err();
        assert!(err.to_string().contains("not initialized"));
    }

    #[test]
    fn completed_packets_are_drained_in_frame_order() {
        let mut pending = std::collections::BTreeMap::new();
        let mut next_frame = 0;

        pending.insert(2, vec![2]);
        assert!(drain_ordered_packets(&mut pending, &mut next_frame).is_empty());

        pending.insert(0, vec![0]);
        assert_eq!(
            drain_ordered_packets(&mut pending, &mut next_frame),
            vec![vec![0]]
        );

        pending.insert(1, vec![1]);
        assert_eq!(
            drain_ordered_packets(&mut pending, &mut next_frame),
            vec![vec![1], vec![2]]
        );
        assert!(pending.is_empty());
        assert_eq!(next_frame, 3);
    }

    #[test]
    fn default_rate_control_settings_are_valid_for_h264() {
        let settings = default_h264_rate_control_settings(
            vk::VideoEncodeRateControlModeFlagsKHR::VBR,
            60,
            None,
        );

        assert!(settings.average_bitrate > 0);
        assert!(settings.max_bitrate >= settings.average_bitrate);
        assert!((0..=51).contains(&settings.min_qp));
        assert!((0..=51).contains(&settings.max_qp));
        assert!(settings.min_qp <= settings.max_qp);
    }

    #[test]
    fn cbr_uses_the_requested_bitrate_for_average_and_peak() {
        let settings = default_h264_rate_control_settings(
            vk::VideoEncodeRateControlModeFlagsKHR::CBR,
            60,
            Some(9_000_000),
        );

        assert_eq!(settings.average_bitrate, 9_000_000);
        assert_eq!(settings.max_bitrate, 9_000_000);
    }

    #[test]
    fn requested_rate_control_mode_is_not_silently_substituted() {
        let vbr_only_caps = vk::VideoEncodeCapabilitiesKHR {
            rate_control_modes: vk::VideoEncodeRateControlModeFlagsKHR::VBR,
            ..Default::default()
        };

        let settings =
            make_h264_rate_control_settings(&vbr_only_caps, 60, None, H264RateControlPolicy::Vbr)
                .unwrap()
                .unwrap();
        assert_eq!(settings.mode, vk::VideoEncodeRateControlModeFlagsKHR::VBR);

        let err =
            make_h264_rate_control_settings(&vbr_only_caps, 60, None, H264RateControlPolicy::Cbr)
                .unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::Unsupported);
    }

    #[test]
    fn frame_and_reference_numbers_wrap_at_the_sps_limit() {
        let sps = make_h264_sps(1920, 1080, None);
        let pps = make_h264_pps();
        assert_eq!(h264_frame_num(15, &sps), 15);
        assert_eq!(h264_frame_num(16, &sps), 0);
        assert_eq!(h264_frame_num(17, &sps), 1);

        let frame = H264FrameInfo::new(17, &sps, &pps, false, 0);
        let reference = make_h264_reference_info(17, false, &sps);
        assert_eq!(frame._picture.frame_num, 1);
        assert_eq!(reference.FrameNum, 1);
        assert_eq!(frame._picture.PicOrderCnt, reference.PicOrderCnt);
    }
}
