use ffmpeg_next::format::{pixel, Pixel};
use ffmpeg_next::{ChannelLayout, StreamMut};

use ffmpeg_next::codec::encoder::{Audio, Video};
use ffmpeg_next::format::context::Output;
use ffmpeg_next::software::scaling;

use crate::video_backend::VideoConfig;
pub struct FfmpegBackend {
    v_enc: Video,
    a_enc: Audio,
    octx: Output,
    v_stream_idx: usize,
    a_stream_idx: usize,
    scaler: scaling::context::Context,
    frame_count: u64,
}

impl FfmpegBackend {
    pub fn new(video_config: &VideoConfig) -> Self {
        ffmpeg_next::init().unwrap();

        // init Muxer
        let mut octx = ffmpeg_next::format::output(&video_config.filename).unwrap();
        let global_header = octx
            .format()
            .flags()
            .contains(ffmpeg_next::format::Flags::GLOBAL_HEADER);
        // config video stream
        let v_codec = ffmpeg_next::encoder::find(ffmpeg_next::codec::Id::H264).unwrap();

        let mut v_stream = octx.add_stream(v_codec).unwrap();
        let v_stream_idx = v_stream.index();

        let mut v_enc_ctx_new = ffmpeg_next::codec::context::Context::new();
        let mut v_enc_ctx = v_enc_ctx_new.encoder().video().unwrap();

        v_enc_ctx.set_width(video_config.output_width);
        v_enc_ctx.set_height(video_config.output_height);
        v_enc_ctx.set_format(pixel::Pixel::RGBAF32LE); // TODO: checks if we need scaling

        v_enc_ctx.set_time_base((1 as i32, video_config.framerate as i32));

        if global_header {
            v_enc_ctx.set_flags(ffmpeg_next::codec::Flags::GLOBAL_HEADER);
        }
        let v_enc = v_enc_ctx.open().unwrap();
        v_stream.set_parameters(&v_enc);

        let a_codec = ffmpeg_next::encoder::find(ffmpeg_next::codec::Id::AAC).unwrap();
        let mut a_stream = octx.add_stream(a_codec).unwrap();
        let a_stream_idx = a_stream.index();
        let mut a_enc_ctx_new = ffmpeg_next::codec::context::Context::new();
        let mut a_enc_ctx = a_enc_ctx_new.encoder().audio().unwrap();
        // 必须设置的 AAC 参数：
        // 1. 采样格式：AAC 通常使用 FLTP (Float Planar)，有些实现也支持 S16
        a_enc_ctx.set_format(ffmpeg_next::format::Sample::F32(
            ffmpeg_next::format::sample::Type::Planar,
        ));

        // 2. 采样率
        a_enc_ctx.set_rate(44100); // 或从 config 获取

        // 3. 声道布局 (立体声)
        a_enc_ctx.set_channel_layout(ChannelLayout::STEREO);
        // set_channels 对于ffmpeg 7.0以后是不存在的

        // 4. 时间基准
        a_enc_ctx.set_time_base((1, 44100));

        // 5. 全局头 (对于 MP4 容器非常重要)
        if global_header {
            a_enc_ctx.set_flags(ffmpeg_next::codec::Flags::GLOBAL_HEADER);
        }

        let a_enc = a_enc_ctx.open_as(a_codec).unwrap();
        a_stream.set_parameters(&a_enc);

        let v_target_format = Pixel::YUV420P;
        let scaler = scaling::context::Context::get(
            Pixel::RGBA, // 输入格式
            video_config.output_width,
            video_config.output_height,
            v_target_format, // 输出格式 (YUV420P)
            video_config.output_width,
            video_config.output_height,
            scaling::flag::Flags::BILINEAR,
        )
        .unwrap();

        octx.write_header().unwrap();

        Self {
            octx,
            v_enc,
            a_enc,
            v_stream_idx,
            a_stream_idx,
            scaler,
            frame_count: 0,
        }
    }

    pub fn write_frame(frame_data: &[u8]) {}
}
