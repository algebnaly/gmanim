//! VAAPI H.264 hardware-accelerated video encoding backend.
//!
//! Public API: [`FfmpegVaapiBackend`] — async encoder with buffer pooling.
//!
//! ## Per-frame pipeline
//!
//!   1. Convert RGBA/BGRA → NV12 (CPU, via `yuv` crate with SIMD)
//!   2. Upload NV12 to GPU surface (`av_hwframe_transfer_data`)
//!   3. Submit to VAAPI H.264 encoder (`avcodec_send_frame`)
//!   4. Mux encoded packets to output file
//!
//! ## Architecture
//!
//! ```text
//!   Main thread                       Worker thread
//!   ──────────                        ─────────────
//!   acquire_buffer() ← recycler ←── return used buffer
//!        │                                  ↑
//!   render into buf                  Encoder::write_frame
//!        │                                  ↑
//!   submit_frame(buf) ── sender ──→ receive buffer
//! ```

use std::ffi::{CStr, CString};
use std::io;
use std::path::Path;
use std::ptr;
use std::sync::mpsc::{Receiver, SyncSender, TryRecvError, sync_channel};
use std::thread::{self, JoinHandle};

use std::{error::Error, fmt};

use ffmpeg_next::ffi;
use yuv::{
    BufferStoreMut, YuvBiPlanarImageMut, YuvConversionMode, YuvRange, YuvStandardMatrix,
    bgra_to_yuv_nv12, rgba_to_yuv_nv12,
};

use crate::video_backend::{ColorOrder, VideoConfig};

// ═══════════════════════════════════════════════════════════════════════════
// FFmpeg error helper
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug)]
struct FfmpegError {
    context: &'static str,
    code: i32,
}

impl fmt::Display for FfmpegError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut buf = [0i8; 256];
        unsafe {
            ffi::av_strerror(self.code, buf.as_mut_ptr(), buf.len());
            let msg = CStr::from_ptr(buf.as_ptr()).to_string_lossy();
            write!(f, "{}: {} ({})", self.context, msg, self.code)
        }
    }
}

impl Error for FfmpegError {}

/// Check an FFmpeg return code, converting negative values to `io::Error`.
fn check(context: &'static str, code: i32) -> io::Result<()> {
    if code < 0 {
        Err(io::Error::other(FfmpegError { context, code }))
    } else {
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Encoder — synchronous, single-threaded H.264 encoder (private)
// ═══════════════════════════════════════════════════════════════════════════

/// Number of pre-allocated// 0 means dynamic allocation. This avoids deadlocks when the encoder holds many frames.
const HW_FRAME_POOL_SIZE: usize = 0;

/// Synchronous VAAPI H.264 encoder (internal implementation detail).
///
/// Used internally by [`FfmpegVaapiBackend`]'s worker thread.
struct Encoder {
    // Muxer / IO
    format_ctx: *mut ffi::AVFormatContext,
    stream: *mut ffi::AVStream,

    // Encoder
    codec_ctx: *mut ffi::AVCodecContext,

    // VAAPI device & frame pool
    device_ref: *mut ffi::AVBufferRef,
    frames_ref: *mut ffi::AVBufferRef,

    // CPU-side staging frame (NV12)
    sw_frame: *mut ffi::AVFrame,
    packet: *mut ffi::AVPacket,

    width: usize,
    height: usize,
    color_order: ColorOrder,

    frame_count: i64,
    in_flight_frames: usize,
    closed: bool,
}

// Safety: all FFmpeg pointers are used exclusively by one thread.
unsafe impl Send for Encoder {}

// --- Construction ---

impl Encoder {
    /// Create a new encoder, panicking on failure.
    fn new(config: &VideoConfig) -> Self {
        Self::try_new(config, "/dev/dri/renderD128").expect("failed to create VAAPI H.264 encoder")
    }

    /// Create a new encoder with an explicit VAAPI device path.
    fn try_new(config: &VideoConfig, vaapi_device: &str) -> io::Result<Self> {
        if config.output_color_profile != crate::OutputColorProfile::Bt709Sdr {
            return Err(io::Error::other(
                "VAAPI H.264 currently supports only 8-bit BT.709 SDR output",
            ));
        }
        ffmpeg_next::init().map_err(io::Error::other)?;

        #[cfg(not(test))]
        ffmpeg_next::log::set_level(ffmpeg_next::log::Level::Quiet);

        validate_config(config)?;

        unsafe { Self::init_pipeline(config, vaapi_device) }
    }

    /// Initialize the full FFmpeg pipeline:
    ///   output file → muxer → encoder → VAAPI device → frame pool.
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn init_pipeline(config: &VideoConfig, vaapi_device: &str) -> io::Result<Self> {
        let output_path = CString::new(Path::new(&config.filename).as_os_str().as_encoded_bytes())
            .map_err(io::Error::other)?;
        let device_path = CString::new(vaapi_device).map_err(io::Error::other)?;

        // Encoder lookup
        let codec = ffi::avcodec_find_encoder_by_name(c"h264_vaapi".as_ptr());
        if codec.is_null() {
            return Err(io::Error::other("h264_vaapi encoder not available"));
        }

        // Muxer
        let mut format_ctx: *mut ffi::AVFormatContext = ptr::null_mut();
        check(
            "avformat_alloc_output_context2",
            ffi::avformat_alloc_output_context2(
                &mut format_ctx,
                ptr::null_mut(),
                c"mp4".as_ptr(),
                output_path.as_ptr(),
            ),
        )?;
        let stream = ffi::avformat_new_stream(format_ctx, ptr::null());
        if stream.is_null() {
            ffi::avformat_free_context(format_ctx);
            return Err(io::Error::other("avformat_new_stream failed"));
        }

        // VAAPI device
        let mut device_ref: *mut ffi::AVBufferRef = ptr::null_mut();
        check(
            "av_hwdevice_ctx_create",
            ffi::av_hwdevice_ctx_create(
                &mut device_ref,
                ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_VAAPI,
                device_path.as_ptr(),
                ptr::null_mut(),
                0,
            ),
        )?;

        // HW frame context
        let frames_ref = ffi::av_hwframe_ctx_alloc(device_ref);
        if frames_ref.is_null() {
            return Err(io::Error::other("av_hwframe_ctx_alloc failed"));
        }
        let frames_ctx = (*frames_ref).data as *mut ffi::AVHWFramesContext;
        (*frames_ctx).format = ffi::AVPixelFormat::AV_PIX_FMT_VAAPI;
        (*frames_ctx).sw_format = ffi::AVPixelFormat::AV_PIX_FMT_NV12;
        (*frames_ctx).width = config.output_width as i32;
        (*frames_ctx).height = config.output_height as i32;
        (*frames_ctx).initial_pool_size = 64; // fixed pool size
        check("av_hwframe_ctx_init", ffi::av_hwframe_ctx_init(frames_ref))?;

        // Codec context
        let codec_ctx = ffi::avcodec_alloc_context3(codec);
        if codec_ctx.is_null() {
            ffi::avformat_free_context(format_ctx);
            return Err(io::Error::other("avcodec_alloc_context3 failed"));
        }
        Self::configure_encoder(codec_ctx, config, format_ctx, frames_ref)?;
        check(
            "avcodec_open2",
            ffi::avcodec_open2(codec_ctx, codec, ptr::null_mut()),
        )?;

        // Wire stream ↔ codec
        (*stream).time_base = (*codec_ctx).time_base;
        check(
            "avcodec_parameters_from_context",
            ffi::avcodec_parameters_from_context((*stream).codecpar, codec_ctx),
        )?;

        // Open output file
        if ((*(*format_ctx).oformat).flags & ffi::AVFMT_NOFILE) == 0 {
            check(
                "avio_open",
                ffi::avio_open(
                    &mut (*format_ctx).pb,
                    output_path.as_ptr(),
                    ffi::AVIO_FLAG_WRITE,
                ),
            )?;
        }
        check(
            "avformat_write_header",
            ffi::avformat_write_header(format_ctx, ptr::null_mut()),
        )?;

        // Allocate frames
        let sw_frame = Self::alloc_sw_frame(config)?;
        let packet = ffi::av_packet_alloc();
        if packet.is_null() {
            return Err(io::Error::other("av_packet_alloc failed"));
        }

        Ok(Self {
            format_ctx,
            stream,
            codec_ctx,
            device_ref,
            frames_ref,
            sw_frame,
            packet,
            width: config.output_width as usize,
            height: config.output_height as usize,
            color_order: config.color_order,
            frame_count: 0,
            in_flight_frames: 0,
            closed: false,
        })
    }

    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn configure_encoder(
        ctx: *mut ffi::AVCodecContext,
        config: &VideoConfig,
        format_ctx: *mut ffi::AVFormatContext,
        frames_ref: *mut ffi::AVBufferRef,
    ) -> io::Result<()> {
        (*ctx).codec_id = ffi::AVCodecID::AV_CODEC_ID_H264;
        (*ctx).codec_type = ffi::AVMediaType::AVMEDIA_TYPE_VIDEO;
        (*ctx).width = config.output_width as i32;
        (*ctx).height = config.output_height as i32;
        (*ctx).pix_fmt = ffi::AVPixelFormat::AV_PIX_FMT_VAAPI;
        (*ctx).time_base = ffi::AVRational {
            num: 1,
            den: config.framerate as i32,
        };
        (*ctx).framerate = ffi::AVRational {
            num: config.framerate as i32,
            den: 1,
        };
        (*ctx).bit_rate = config.bitrate.unwrap_or(2_000_000) as i64;
        (*ctx).gop_size = config.framerate as i32;
        (*ctx).max_b_frames = 0;
        (*ctx).color_range = ffi::AVColorRange::AVCOL_RANGE_MPEG;
        (*ctx).colorspace = ffi::AVColorSpace::AVCOL_SPC_BT709;
        (*ctx).color_primaries = ffi::AVColorPrimaries::AVCOL_PRI_BT709;
        (*ctx).color_trc = ffi::AVColorTransferCharacteristic::AVCOL_TRC_BT709;
        (*ctx).hw_frames_ctx = ffi::av_buffer_ref(frames_ref);
        if (*ctx).hw_frames_ctx.is_null() {
            return Err(io::Error::other("av_buffer_ref for hw_frames_ctx failed"));
        }
        if ((*(*format_ctx).oformat).flags & ffi::AVFMT_GLOBALHEADER) != 0 {
            (*ctx).flags |= ffi::AV_CODEC_FLAG_GLOBAL_HEADER as i32;
        }
        Ok(())
    }

    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn alloc_sw_frame(config: &VideoConfig) -> io::Result<*mut ffi::AVFrame> {
        let frame = ffi::av_frame_alloc();
        if frame.is_null() {
            return Err(io::Error::other("av_frame_alloc for sw_frame failed"));
        }
        (*frame).format = ffi::AVPixelFormat::AV_PIX_FMT_NV12 as i32;
        (*frame).width = config.output_width as i32;
        (*frame).height = config.output_height as i32;
        (*frame).color_range = ffi::AVColorRange::AVCOL_RANGE_MPEG;
        (*frame).colorspace = ffi::AVColorSpace::AVCOL_SPC_BT709;
        (*frame).color_primaries = ffi::AVColorPrimaries::AVCOL_PRI_BT709;
        (*frame).color_trc = ffi::AVColorTransferCharacteristic::AVCOL_TRC_BT709;
        check("av_frame_get_buffer", ffi::av_frame_get_buffer(frame, 32))?;
        Ok(frame)
    }
}

// --- Frame encoding ---

impl Encoder {
    /// Encode one RGBA/BGRA frame (convert → upload → encode, synchronously).
    #[allow(unsafe_op_in_unsafe_fn)]
    fn write_frame(&mut self, frame_data: &[u8]) {
        unsafe {
            // Prevent deadlock by limiting in-flight frames to 10
            while self.in_flight_frames >= 10 {
                if self.drain_packets_once().unwrap() {
                    continue;
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
            }

            // 1. Convert RGBA → NV12
            check(
                "av_frame_make_writable",
                ffi::av_frame_make_writable(self.sw_frame),
            )
            .unwrap();

            convert_rgba_to_nv12(
                self.sw_frame,
                frame_data,
                self.width,
                self.height,
                self.color_order,
            );
            (*self.sw_frame).pts = self.frame_count;

            // 2. Allocate hardware frame
            let mut hw_frame = ffi::av_frame_alloc();
            (*hw_frame).format = ffi::AVPixelFormat::AV_PIX_FMT_VAAPI as i32;
            (*hw_frame).hw_frames_ctx = ffi::av_buffer_ref(self.frames_ref);
            check(
                "av_hwframe_get_buffer",
                ffi::av_hwframe_get_buffer(self.frames_ref, hw_frame, 0),
            )
            .unwrap();

            // 3. Upload NV12 to VAAPI surface
            check(
                "av_hwframe_transfer_data",
                ffi::av_hwframe_transfer_data(hw_frame, self.sw_frame, 0),
            )
            .unwrap();

            // 4. Encode
            (*hw_frame).pts = self.frame_count;
            loop {
                let ret = ffi::avcodec_send_frame(self.codec_ctx, hw_frame);
                if ret == ffi::AVERROR(ffi::EAGAIN) {
                    if !self.drain_packets_once().unwrap() {
                        std::thread::sleep(std::time::Duration::from_millis(1));
                    }
                    continue;
                }
                check("avcodec_send_frame", ret).unwrap();
                break;
            }

            self.in_flight_frames += 1;
            ffi::av_frame_free(&mut hw_frame);
            self.drain_packets().unwrap();
            self.frame_count += 1;
        }
    }

    /// Flush the encoder and write the file trailer.
    #[allow(unsafe_op_in_unsafe_fn)]
    fn finish(&mut self) -> io::Result<()> {
        if self.closed {
            return Ok(());
        }
        unsafe {
            check(
                "avcodec_send_frame (flush)",
                ffi::avcodec_send_frame(self.codec_ctx, ptr::null_mut()),
            )
            .unwrap();
            self.drain_packets().unwrap();
            check("av_write_trailer", ffi::av_write_trailer(self.format_ctx))?;
        }
        self.closed = true;

        Ok(())
    }

    /// Drain all available encoded packets from the encoder and write to muxer.
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn drain_packets_once(&mut self) -> io::Result<bool> {
        let ret = ffi::avcodec_receive_packet(self.codec_ctx, self.packet);
        if ret == ffi::AVERROR(ffi::EAGAIN) || ret == ffi::AVERROR_EOF {
            return Ok(false);
        }
        check("avcodec_receive_packet", ret)?;

        ffi::av_packet_rescale_ts(
            self.packet,
            (*self.codec_ctx).time_base,
            (*self.stream).time_base,
        );
        (*self.packet).stream_index = (*self.stream).index;
        // eprintln!("drain_packets_once: av_interleaved_write_frame");
        check(
            "av_interleaved_write_frame",
            ffi::av_interleaved_write_frame(self.format_ctx, self.packet),
        )?;
        // eprintln!("drain_packets_once: av_packet_unref");
        ffi::av_packet_unref(self.packet);

        if self.in_flight_frames > 0 {
            self.in_flight_frames -= 1;
        }
        Ok(true)
    }

    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn drain_packets(&mut self) -> io::Result<()> {
        while self.drain_packets_once()? {}
        Ok(())
    }
}

// --- Cleanup ---

impl Drop for Encoder {
    fn drop(&mut self) {
        let _ = self.finish();
        unsafe {
            if !self.packet.is_null() {
                ffi::av_packet_free(&mut self.packet);
            }
            if !self.sw_frame.is_null() {
                ffi::av_frame_free(&mut self.sw_frame);
            }
            if !self.frames_ref.is_null() {
                ffi::av_buffer_unref(&mut self.frames_ref);
            }
            if !self.device_ref.is_null() {
                ffi::av_buffer_unref(&mut self.device_ref);
            }
            if !self.codec_ctx.is_null() {
                ffi::avcodec_free_context(&mut self.codec_ctx);
            }
            if !self.format_ctx.is_null() {
                if !(*self.format_ctx).pb.is_null()
                    && ((*(*self.format_ctx).oformat).flags & ffi::AVFMT_NOFILE) == 0
                {
                    ffi::avio_closep(&mut (*self.format_ctx).pb);
                }
                ffi::avformat_free_context(self.format_ctx);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// FfmpegVaapiBackend — public async backend with buffer pooling
// ═══════════════════════════════════════════════════════════════════════════

use super::FrameBuffer;

enum WorkerMessage {
    Frame(Vec<u8>),
    Finish,
}

/// VAAPI H.264 encoding backend with async worker thread and buffer pooling.
///
/// Wraps an internal encoder in a worker thread with a bounded frame queue.
/// Eliminates per-frame allocation by recycling buffers between the main
/// thread and the encoding worker.
///
/// ## Usage
///
/// **Zero-copy (preferred):**
/// ```ignore
/// let mut buf = backend.acquire_buffer();
/// render_into(buf.as_mut_slice());
/// backend.submit_frame(buf);
/// ```
///
/// **Compatibility:**
/// ```ignore
/// backend.write_frame(&rgba_bytes);
/// ```
pub struct FfmpegVaapiBackend {
    sender: SyncSender<WorkerMessage>,
    recycler: Receiver<Vec<u8>>,
    worker: Option<JoinHandle<io::Result<()>>>,
    frame_size: usize,
}

impl FfmpegVaapiBackend {
    /// Create with default queue depth (3 frames in flight).
    pub fn new(config: &VideoConfig) -> Self {
        Self::try_new(config, 3).expect("failed to create VAAPI backend")
    }

    /// Create with configurable queue depth.
    pub fn try_new(config: &VideoConfig, queue_depth: usize) -> io::Result<Self> {
        let config_clone = config.clone();
        let frame_size = match config.color_order {
            ColorOrder::Nv12 => {
                let y_size = config.output_width as usize * config.output_height as usize;
                let uv_size = config.output_width as usize * config.output_height as usize / 2;
                y_size + uv_size
            }
            ColorOrder::Yuv444p => config.output_width as usize * config.output_height as usize * 3,
            _ => config.output_width as usize * config.output_height as usize * 4,
        };
        let encoder = Encoder::try_new(&config_clone, "/dev/dri/renderD128")?;

        let (sender, receiver) = sync_channel(queue_depth);
        let (recycle_tx, recycle_rx) = sync_channel(queue_depth + 1);

        let worker = thread::spawn(move || {
            let mut encoder = encoder;
            let mut frames_processed = 0;
            while let Ok(msg) = receiver.recv() {
                match msg {
                    WorkerMessage::Frame(buf) => {
                        frames_processed += 1;
                        encoder.write_frame(&buf);
                        let _ = recycle_tx.try_send(buf); // recycle to pool if space, otherwise drop
                    }
                    WorkerMessage::Finish => {
                        break;
                    }
                }
            }
            encoder.finish()
        });

        Ok(Self {
            sender,
            recycler: recycle_rx,
            worker: Some(worker),
            frame_size,
        })
    }

    /// Get a buffer to render into (recycled or newly allocated).
    pub fn acquire_buffer(&mut self) -> FrameBuffer {
        let data = match self.recycler.try_recv() {
            Ok(buf) => {
                debug_assert_eq!(buf.len(), self.frame_size);
                buf
            }
            Err(TryRecvError::Empty | TryRecvError::Disconnected) => {
                vec![0u8; self.frame_size]
            }
        };
        FrameBuffer { data }
    }

    /// Submit a rendered frame for encoding — zero copy, ownership transfer.
    pub fn submit_frame(&mut self, buf: FrameBuffer) {
        debug_assert_eq!(buf.data.len(), self.frame_size);
        self.sender
            .send(WorkerMessage::Frame(buf.data))
            .expect("encoding worker has stopped unexpectedly");
    }

    /// Compatibility path: copies data into a pooled buffer then submits.
    pub fn write_frame(&mut self, frame_data: &[u8]) {
        let mut buf = self.acquire_buffer();
        buf.data.copy_from_slice(frame_data);
        self.submit_frame(buf);
    }

    /// Flush all queued frames and finalize the output file.
    pub fn finish(&mut self) -> io::Result<()> {
        let _ = self.sender.send(WorkerMessage::Finish);
        if let Some(handle) = self.worker.take() {
            handle
                .join()
                .map_err(|_| io::Error::other("encoding worker panicked"))??;
        }
        Ok(())
    }
}

impl Drop for FfmpegVaapiBackend {
    fn drop(&mut self) {
        let _ = self.finish();
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Color conversion helper
// ═══════════════════════════════════════════════════════════════════════════

/// Convert a packed RGBA/BGRA buffer into NV12 planes of an `AVFrame`.
fn convert_rgba_to_nv12(
    frame: *mut ffi::AVFrame,
    rgba: &[u8],
    width: usize,
    height: usize,
    color_order: ColorOrder,
) {
    unsafe {
        let y_stride = (*frame).linesize[0] as usize;
        let uv_stride = (*frame).linesize[1] as usize;
        let y_plane = std::slice::from_raw_parts_mut((*frame).data[0], y_stride * height);
        let uv_plane = std::slice::from_raw_parts_mut((*frame).data[1], uv_stride * height / 2);

        let mut image = YuvBiPlanarImageMut {
            y_plane: BufferStoreMut::Borrowed(y_plane),
            y_stride: y_stride as u32,
            uv_plane: BufferStoreMut::Borrowed(uv_plane),
            uv_stride: uv_stride as u32,
            width: width as u32,
            height: height as u32,
        };

        if let ColorOrder::Nv12 = color_order {
            let src_y = &rgba[..(width * height) as usize];
            let src_uv = &rgba[(width * height) as usize..];
            for (dst_row, src_row) in y_plane
                .chunks_exact_mut(y_stride)
                .zip(src_y.chunks_exact(width as usize))
            {
                dst_row[..width as usize].copy_from_slice(src_row);
            }
            for (dst_row, src_row) in uv_plane
                .chunks_exact_mut(uv_stride)
                .zip(src_uv.chunks_exact(width as usize))
            {
                dst_row[..width as usize].copy_from_slice(src_row);
            }
            return;
        }

        let convert = match color_order {
            ColorOrder::Rgba => rgba_to_yuv_nv12,
            ColorOrder::Bgra => bgra_to_yuv_nv12,
            ColorOrder::Nv12 | ColorOrder::Yuv444p => unreachable!(),
        };
        convert(
            &mut image,
            rgba,
            (width * 4) as u32,
            YuvRange::Limited,
            YuvStandardMatrix::Bt709,
            YuvConversionMode::Fast,
        )
        .expect("RGBA/BGRA → NV12 conversion failed");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Config validation
// ═══════════════════════════════════════════════════════════════════════════

fn validate_config(config: &VideoConfig) -> io::Result<()> {
    if config.output_width == 0 || config.output_height == 0 {
        return Err(io::Error::other("video dimensions must be positive"));
    }
    if config.output_width % 2 != 0 || config.output_height % 2 != 0 {
        return Err(io::Error::other(
            "VAAPI H.264 requires even width and height",
        ));
    }
    if config.framerate == 0 {
        return Err(io::Error::other("framerate must be positive"));
    }
    Ok(())
}
