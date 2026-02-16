use ffmpeg_next::format::pixel;

use crate::video_backend::VideoConfig;
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