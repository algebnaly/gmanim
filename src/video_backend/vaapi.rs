use std::error::Error;
use std::ffi::{CStr, CString};
use std::fmt;
use std::io;
use std::path::Path;
use std::ptr;

use ffmpeg_next::ffi;
use yuv::{
    bgra_to_yuv_nv12, rgba_to_yuv_nv12, BufferStoreMut, YuvBiPlanarImageMut, YuvConversionMode,
    YuvRange, YuvStandardMatrix,
};

use crate::video_backend::{ColorOrder, VideoConfig};

const HW_FRAME_POOL_SIZE: usize = 16;

pub struct FfmpegVaapiH264Backend {
    format_ctx: *mut ffi::AVFormatContext,
    codec_ctx: *mut ffi::AVCodecContext,
    stream: *mut ffi::AVStream,
    device_ref: *mut ffi::AVBufferRef,
    frames_ref: *mut ffi::AVBufferRef,
    sw_frame: *mut ffi::AVFrame,
    hw_frames: Vec<*mut ffi::AVFrame>,
    packet: *mut ffi::AVPacket,
    next_hw_frame: usize,
    width: usize,
    height: usize,
    color_order: ColorOrder,
    frame_count: i64,
    closed: bool,
}

impl FfmpegVaapiH264Backend {
    pub fn new(video_config: &VideoConfig) -> Self {
        Self::try_new(video_config, "/dev/dri/renderD128")
            .expect("failed to create VAAPI H.264 backend")
    }

    pub fn try_new(video_config: &VideoConfig, vaapi_device: &str) -> io::Result<Self> {
        ffmpeg_next::init().map_err(io::Error::other)?;

        #[cfg(not(test))]
        ffmpeg_next::log::set_level(ffmpeg_next::log::Level::Quiet);

        validate_vaapi_config(video_config)?;

        unsafe { Self::try_new_inner(video_config, vaapi_device) }
    }

    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn try_new_inner(video_config: &VideoConfig, vaapi_device: &str) -> io::Result<Self> {
        let output_c = CString::new(
            Path::new(&video_config.filename)
                .as_os_str()
                .as_encoded_bytes(),
        )
        .map_err(io::Error::other)?;
        let device_c = CString::new(vaapi_device).map_err(io::Error::other)?;
        let encoder_c = CString::new("h264_vaapi").unwrap();
        let format_c = CString::new("mp4").unwrap();

        let codec = ffi::avcodec_find_encoder_by_name(encoder_c.as_ptr());
        if codec.is_null() {
            return Err(io::Error::other(
                "FFmpeg encoder h264_vaapi is not available",
            ));
        }

        let mut format_ctx: *mut ffi::AVFormatContext = ptr::null_mut();
        check(
            "avformat_alloc_output_context2",
            ffi::avformat_alloc_output_context2(
                &mut format_ctx,
                ptr::null_mut(),
                format_c.as_ptr(),
                output_c.as_ptr(),
            ),
        )?;

        let stream = ffi::avformat_new_stream(format_ctx, ptr::null());
        if stream.is_null() {
            ffi::avformat_free_context(format_ctx);
            return Err(io::Error::other("avformat_new_stream failed"));
        }

        let codec_ctx = ffi::avcodec_alloc_context3(codec);
        if codec_ctx.is_null() {
            ffi::avformat_free_context(format_ctx);
            return Err(io::Error::other("avcodec_alloc_context3 failed"));
        }

        let mut device_ref: *mut ffi::AVBufferRef = ptr::null_mut();
        check(
            "av_hwdevice_ctx_create",
            ffi::av_hwdevice_ctx_create(
                &mut device_ref,
                ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_VAAPI,
                device_c.as_ptr(),
                ptr::null_mut(),
                0,
            ),
        )?;

        let frames_ref = ffi::av_hwframe_ctx_alloc(device_ref);
        if frames_ref.is_null() {
            return Err(io::Error::other("av_hwframe_ctx_alloc failed"));
        }
        {
            let frames_ctx = (*frames_ref).data as *mut ffi::AVHWFramesContext;
            (*frames_ctx).format = ffi::AVPixelFormat::AV_PIX_FMT_VAAPI;
            (*frames_ctx).sw_format = ffi::AVPixelFormat::AV_PIX_FMT_NV12;
            (*frames_ctx).width = video_config.output_width as i32;
            (*frames_ctx).height = video_config.output_height as i32;
            (*frames_ctx).initial_pool_size = HW_FRAME_POOL_SIZE as i32;
        }
        check("av_hwframe_ctx_init", ffi::av_hwframe_ctx_init(frames_ref))?;

        (*codec_ctx).codec_id = ffi::AVCodecID::AV_CODEC_ID_H264;
        (*codec_ctx).codec_type = ffi::AVMediaType::AVMEDIA_TYPE_VIDEO;
        (*codec_ctx).width = video_config.output_width as i32;
        (*codec_ctx).height = video_config.output_height as i32;
        (*codec_ctx).pix_fmt = ffi::AVPixelFormat::AV_PIX_FMT_VAAPI;
        (*codec_ctx).time_base = ffi::AVRational {
            num: 1,
            den: video_config.framerate as i32,
        };
        (*codec_ctx).framerate = ffi::AVRational {
            num: video_config.framerate as i32,
            den: 1,
        };
        (*codec_ctx).bit_rate = 2_000_000;
        (*codec_ctx).gop_size = video_config.framerate as i32;
        (*codec_ctx).max_b_frames = 0;
        (*codec_ctx).hw_frames_ctx = ffi::av_buffer_ref(frames_ref);
        if (*codec_ctx).hw_frames_ctx.is_null() {
            return Err(io::Error::other("av_buffer_ref for hw_frames_ctx failed"));
        }
        if ((*(*format_ctx).oformat).flags & ffi::AVFMT_GLOBALHEADER) != 0 {
            (*codec_ctx).flags |= ffi::AV_CODEC_FLAG_GLOBAL_HEADER as i32;
        }

        check(
            "avcodec_open2",
            ffi::avcodec_open2(codec_ctx, codec, ptr::null_mut()),
        )?;

        (*stream).time_base = (*codec_ctx).time_base;
        check(
            "avcodec_parameters_from_context",
            ffi::avcodec_parameters_from_context((*stream).codecpar, codec_ctx),
        )?;

        if ((*(*format_ctx).oformat).flags & ffi::AVFMT_NOFILE) == 0 {
            check(
                "avio_open",
                ffi::avio_open(
                    &mut (*format_ctx).pb,
                    output_c.as_ptr(),
                    ffi::AVIO_FLAG_WRITE,
                ),
            )?;
        }

        check(
            "avformat_write_header",
            ffi::avformat_write_header(format_ctx, ptr::null_mut()),
        )?;

        let sw_frame = ffi::av_frame_alloc();
        let packet = ffi::av_packet_alloc();
        if sw_frame.is_null() || packet.is_null() {
            return Err(io::Error::other(
                "failed to allocate FFmpeg frame or packet",
            ));
        }

        (*sw_frame).format = ffi::AVPixelFormat::AV_PIX_FMT_NV12 as i32;
        (*sw_frame).width = video_config.output_width as i32;
        (*sw_frame).height = video_config.output_height as i32;
        check(
            "av_frame_get_buffer",
            ffi::av_frame_get_buffer(sw_frame, 32),
        )?;

        let mut hw_frames = Vec::with_capacity(HW_FRAME_POOL_SIZE);
        for _ in 0..HW_FRAME_POOL_SIZE {
            let hw_frame = ffi::av_frame_alloc();
            if hw_frame.is_null() {
                return Err(io::Error::other("av_frame_alloc for VAAPI frame failed"));
            }

            (*hw_frame).format = ffi::AVPixelFormat::AV_PIX_FMT_VAAPI as i32;
            (*hw_frame).width = video_config.output_width as i32;
            (*hw_frame).height = video_config.output_height as i32;
            (*hw_frame).hw_frames_ctx = ffi::av_buffer_ref(frames_ref);
            if (*hw_frame).hw_frames_ctx.is_null() {
                return Err(io::Error::other(
                    "av_buffer_ref for frame hw_frames_ctx failed",
                ));
            }
            check(
                "av_hwframe_get_buffer",
                ffi::av_hwframe_get_buffer(frames_ref, hw_frame, 0),
            )?;
            hw_frames.push(hw_frame);
        }

        Ok(Self {
            format_ctx,
            codec_ctx,
            stream,
            device_ref,
            frames_ref,
            sw_frame,
            hw_frames,
            packet,
            next_hw_frame: 0,
            width: video_config.output_width as usize,
            height: video_config.output_height as usize,
            color_order: video_config.color_order,
            frame_count: 0,
            closed: false,
        })
    }

    #[allow(unsafe_op_in_unsafe_fn)]
    pub fn write_frame(&mut self, frame_data: &[u8]) {
        assert_eq!(frame_data.len(), self.width * self.height * 4);

        unsafe {
            check(
                "av_frame_make_writable",
                ffi::av_frame_make_writable(self.sw_frame),
            )
            .unwrap();
            fill_nv12_from_packed_rgba(
                self.sw_frame,
                frame_data,
                self.width,
                self.height,
                self.color_order,
            );
            (*self.sw_frame).pts = self.frame_count;

            let hw_frame = self.hw_frames[self.next_hw_frame];
            self.next_hw_frame = (self.next_hw_frame + 1) % self.hw_frames.len();
            check(
                "av_frame_make_writable hw",
                ffi::av_frame_make_writable(hw_frame),
            )
            .unwrap();
            check(
                "av_hwframe_transfer_data",
                ffi::av_hwframe_transfer_data(hw_frame, self.sw_frame, 0),
            )
            .unwrap();
            (*hw_frame).pts = self.frame_count;

            check(
                "avcodec_send_frame",
                ffi::avcodec_send_frame(self.codec_ctx, hw_frame),
            )
            .unwrap();
            self.write_available_packets().unwrap();

            self.frame_count += 1;
        }
    }

    #[allow(unsafe_op_in_unsafe_fn)]
    pub fn finish(&mut self) -> io::Result<()> {
        if self.closed {
            return Ok(());
        }

        unsafe {
            check(
                "avcodec_send_frame flush",
                ffi::avcodec_send_frame(self.codec_ctx, ptr::null()),
            )?;
            self.write_available_packets()?;
            check("av_write_trailer", ffi::av_write_trailer(self.format_ctx))?;
        }
        self.closed = true;
        Ok(())
    }

    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn write_available_packets(&mut self) -> io::Result<()> {
        loop {
            let ret = ffi::avcodec_receive_packet(self.codec_ctx, self.packet);
            if ret == ffi::AVERROR(ffi::EAGAIN) || ret == ffi::AVERROR_EOF {
                return Ok(());
            }
            check("avcodec_receive_packet", ret)?;

            ffi::av_packet_rescale_ts(
                self.packet,
                (*self.codec_ctx).time_base,
                (*self.stream).time_base,
            );
            (*self.packet).stream_index = (*self.stream).index;
            check(
                "av_interleaved_write_frame",
                ffi::av_interleaved_write_frame(self.format_ctx, self.packet),
            )?;
            ffi::av_packet_unref(self.packet);
        }
    }
}

impl Drop for FfmpegVaapiH264Backend {
    fn drop(&mut self) {
        let _ = self.finish();
        unsafe {
            if !self.packet.is_null() {
                ffi::av_packet_free(&mut self.packet);
            }
            for mut hw_frame in self.hw_frames.drain(..) {
                ffi::av_frame_free(&mut hw_frame);
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

fn validate_vaapi_config(video_config: &VideoConfig) -> io::Result<()> {
    if video_config.output_width == 0 || video_config.output_height == 0 {
        return Err(io::Error::other("video dimensions must be positive"));
    }
    if video_config.output_width % 2 != 0 || video_config.output_height % 2 != 0 {
        return Err(io::Error::other(
            "VAAPI H.264 backend requires even width and height",
        ));
    }
    if video_config.framerate == 0 {
        return Err(io::Error::other("framerate must be positive"));
    }
    Ok(())
}

fn fill_nv12_from_packed_rgba(
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

        let result = match color_order {
            ColorOrder::Rgba => rgba_to_yuv_nv12(
                &mut image,
                rgba,
                (width * 4) as u32,
                YuvRange::Limited,
                YuvStandardMatrix::Bt601,
                YuvConversionMode::Fast,
            ),
            ColorOrder::Bgra => bgra_to_yuv_nv12(
                &mut image,
                rgba,
                (width * 4) as u32,
                YuvRange::Limited,
                YuvStandardMatrix::Bt601,
                YuvConversionMode::Fast,
            ),
        };
        result.expect("failed to convert packed RGBA/BGRA frame to NV12");
    }
}

#[derive(Debug)]
struct FfmpegVaapiError {
    context: &'static str,
    code: i32,
}

impl fmt::Display for FfmpegVaapiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut buf = [0i8; 256];
        unsafe {
            ffi::av_strerror(self.code, buf.as_mut_ptr(), buf.len());
            let msg = CStr::from_ptr(buf.as_ptr()).to_string_lossy();
            write!(f, "{}: {} ({})", self.context, msg, self.code)
        }
    }
}

impl Error for FfmpegVaapiError {}

fn check(context: &'static str, code: i32) -> io::Result<()> {
    if code < 0 {
        Err(io::Error::other(FfmpegVaapiError { context, code }))
    } else {
        Ok(())
    }
}
