use std::io::Write;

use ffmpeg_next::format::{pixel, Pixel};
use ffmpeg_next::Dictionary;
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

        // #[cfg(test)]
        // ffmpeg_next::log::set_level(ffmpeg_next::log::Level::Quiet);

        // 1. Init Muxer
        let mut octx = ffmpeg_next::format::output(&video_config.filename).unwrap();
        let global_header = octx
            .format()
            .flags()
            .contains(ffmpeg_next::format::Flags::GLOBAL_HEADER);

        // ==========================================
        //  视频流配置 (Video Stream)
        // ==========================================
        let v_codec = ffmpeg_next::encoder::find_by_name("libx264")
            .or_else(|| ffmpeg_next::encoder::find(ffmpeg_next::codec::Id::H264))
            .expect("H.264 encoder not found");

        let mut v_stream = octx.add_stream(v_codec).unwrap();
        let v_stream_idx = v_stream.index();

        let mut v_enc_ctx = ffmpeg_next::codec::context::Context::new();
        let mut v_enc = v_enc_ctx.encoder().video().unwrap();

        v_enc.set_width(video_config.output_width);
        v_enc.set_height(video_config.output_height);
        v_enc.set_format(Pixel::YUV420P);
        v_enc.set_time_base((1, video_config.framerate as i32));
        v_enc.set_gop(10);
        v_enc.set_max_b_frames(1);

        v_enc.set_qmin(10);
        v_enc.set_qmax(51);

        if global_header {
            v_enc.set_flags(ffmpeg_next::codec::Flags::GLOBAL_HEADER);
        }

        let mut v_opts = Dictionary::new();

        v_opts.set("preset", "medium");
        v_opts.set("crf", "23");
        v_opts.set("profile", "high");

        let v_enc = v_enc
            .open_as_with(v_codec, v_opts)
            .expect("Failed to open libx264");
        v_stream.set_parameters(&v_enc);

        // ==========================================
        //  音频流配置
        // ==========================================
        let a_codec = ffmpeg_next::encoder::find(ffmpeg_next::codec::Id::AAC).unwrap();
        let mut a_stream = octx.add_stream(a_codec).unwrap();
        let a_stream_idx = a_stream.index();

        let mut a_enc_ctx = ffmpeg_next::codec::context::Context::new();
        let mut a_enc = a_enc_ctx.encoder().audio().unwrap();

        a_enc.set_format(ffmpeg_next::format::Sample::F32(
            ffmpeg_next::format::sample::Type::Planar,
        ));
        a_enc.set_rate(44100);
        a_enc.set_channel_layout(ChannelLayout::STEREO);
        a_enc.set_time_base((1, 44100));

        if global_header {
            a_enc.set_flags(ffmpeg_next::codec::Flags::GLOBAL_HEADER);
        }

        let a_enc = a_enc.open_as(a_codec).unwrap();
        a_stream.set_parameters(&a_enc);

        let scaler = scaling::context::Context::get(
            Pixel::RGBA,
            video_config.output_width,
            video_config.output_height,
            Pixel::YUV420P,
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

    pub fn write_frame(&mut self, frame_data: &[u8]) {
        
        let width = self.v_enc.width();
        let height = self.v_enc.height();
        let mut input_frame = ffmpeg_next::util::frame::video::Video::empty();
        unsafe {
            input_frame.alloc(pixel::Pixel::RGBA, width, height);
        }

        // 2. 填充数据
        // frame_data 必须是 RGBA packed 格式 (长度 = width * height * 4)
        let stride = (self.v_enc.width() * 4) as usize;

        unsafe {
            let mut data = input_frame.data_mut(0);
            data.write(frame_data);
        }

        // 3. 创建输出 frame (YUV420P)
        let mut output_frame = ffmpeg_next::util::frame::video::Video::empty();
        unsafe {
            output_frame.alloc(Pixel::YUV420P, width, height);
        }

        // 4. 转换格式 (RGBA -> YUV420P)
        self.scaler.run(&input_frame, &mut output_frame).unwrap(); // TODO: need measure time here

        // 5. 设置 PTS (Presentation Time Stamp)
        output_frame.set_pts(Some(self.frame_count as i64));
        self.frame_count += 1;

        // 6. 发送给编码器
        self.send_frame(&output_frame);
    }

    // 辅助函数：发送并接收包
    fn send_frame(&mut self, frame: &ffmpeg_next::util::frame::video::Video) {
        self.v_enc.send_frame(frame).unwrap();

        loop {
            let mut packet = ffmpeg_next::Packet::empty();
            match self.v_enc.receive_packet(&mut packet) {
                Ok(_) => {
                    packet.set_stream(self.v_stream_idx);
                    // 重要：转换时间基
                    packet.rescale_ts(
                        self.v_enc.time_base(),
                        self.octx.stream(self.v_stream_idx).unwrap().time_base(),
                    );
                    packet.write_interleaved(&mut self.octx).unwrap();
                }
                Err(e) => break, // EAGAIN or EOF
            }
        }
    }
}
