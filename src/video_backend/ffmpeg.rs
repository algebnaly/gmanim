use std::hint::black_box;
use std::io::Write;
use std::time::Instant;

use ffmpeg_next::format::{pixel, Pixel};
use ffmpeg_next::Dictionary;
use ffmpeg_next::{ChannelLayout, StreamMut};

use ffmpeg_next::codec::encoder::{Audio, Video};
use ffmpeg_next::format::context::Output;
use ffmpeg_next::software::scaling;
use yuv::rgba_to_yuv420;

use crate::video_backend::VideoConfig;
pub struct FfmpegBackend {
    v_enc: Video,
    a_enc: Audio,
    octx: Output,
    v_stream_idx: usize,
    a_stream_idx: usize,
    // scaler: scaling::context::Context,
    frame_count: u64,
}

impl FfmpegBackend {
    pub fn new(video_config: &VideoConfig) -> Self {
        ffmpeg_next::init().unwrap();

        #[cfg(not(test))]
        ffmpeg_next::log::set_level(ffmpeg_next::log::Level::Quiet);

        let mut octx = ffmpeg_next::format::output(&video_config.filename).unwrap();
        let global_header = octx
            .format()
            .flags()
            .contains(ffmpeg_next::format::Flags::GLOBAL_HEADER);

        // video codec settings
        let v_codec = ffmpeg_next::encoder::find(ffmpeg_next::codec::Id::H264)
            .expect("H.264 encoder not found");

        let mut v_stream = octx.add_stream(v_codec).unwrap();
        let v_stream_idx = v_stream.index();

        let mut v_enc_ctx = ffmpeg_next::codec::context::Context::new_with_codec(v_codec);
        let mut v_enc = v_enc_ctx.encoder().video().unwrap();

        v_enc.set_width(video_config.output_width);
        v_enc.set_height(video_config.output_height);
        v_enc.set_format(Pixel::YUV420P);
        v_enc.set_time_base((1, video_config.framerate as i32));
        v_enc.set_gop(12);

        if global_header {
            v_enc.set_flags(ffmpeg_next::codec::Flags::GLOBAL_HEADER);
        }

        let mut v_opts = Dictionary::new();
        v_opts.set("preset", "ultrafast");
        v_opts.set("tune", "fastdecode");

        let v_enc = v_enc
            .open_as_with(v_codec, v_opts)
            .expect("Failed to open libx264");
        v_stream.set_parameters(&v_enc);

        // audio codec settings
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

        octx.write_header().unwrap();

        Self {
            octx,
            v_enc,
            a_enc,
            v_stream_idx,
            a_stream_idx,
            // scaler,
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

        let stride = (self.v_enc.width() * 4) as usize;

        unsafe {
            let mut data = input_frame.data_mut(0);
            data.copy_from_slice(frame_data); // assume no padding needed
        }

        let mut output_frame = ffmpeg_next::util::frame::video::Video::empty();
        unsafe {
            output_frame.alloc(Pixel::YUV420P, width, height);
        }
        do_scale(&input_frame, &mut output_frame);
        // self.scaler.run(&input_frame, &mut output_frame).unwrap(); // TODO: need measure time here
        let d = output_frame.data(0);

        output_frame.set_pts(Some(self.frame_count as i64));
        self.frame_count += 1;

        self.send_frame(&output_frame);
    }

    fn send_frame(&mut self, frame: &ffmpeg_next::util::frame::video::Video) {
        self.v_enc.send_frame(frame).unwrap();
        self.write_video_packet();
    }

    pub fn finish(&mut self) {
        self.v_enc.send_eof().unwrap();
        self.write_video_packet();
        self.octx.write_trailer().unwrap();
    }

    // before call this function, send_frame to encoder first
    fn write_video_packet(&mut self) {
        loop {
            let mut packet = ffmpeg_next::Packet::empty();
            match self.v_enc.receive_packet(&mut packet) {
                Ok(_) => {
                    packet.set_stream(self.v_stream_idx);

                    // since codec time base is different from container's time base,
                    // we need rescale time scale before writing to container
                    packet.rescale_ts(
                        self.v_enc.time_base(),
                        self.octx.stream(self.v_stream_idx).unwrap().time_base(),
                    );
                    packet.write_interleaved(&mut self.octx).unwrap();
                }
                Err(e) => {
                    break;
                } // EAGAIN or EOF
            }
        }
    }
}

fn do_scale(
    input_frame: &ffmpeg_next::util::frame::Video,
    output_frame: &mut ffmpeg_next::util::frame::Video,
) {
    use yuv::rgba_to_yuv420;
    use yuv::BufferStoreMut;
    use yuv::YuvConversionMode;
    use yuv::YuvPlanarImageMut;
    use yuv::YuvRange;
    use yuv::YuvStandardMatrix;

    let y_stride = output_frame.plane_width(0);
    let u_stride = output_frame.plane_width(1);
    let v_stride = output_frame.plane_width(2);
    let width = output_frame.width();
    let height = output_frame.height();

    // 用 unsafe 分别获取各平面的可变指针
    let (y_plane, u_plane, v_plane) = unsafe {
        let ptr = output_frame.as_mut_ptr();
        let y = std::slice::from_raw_parts_mut((*ptr).data[0], (y_stride * height) as usize);
        let u = std::slice::from_raw_parts_mut((*ptr).data[1], (u_stride * height / 2) as usize);
        let v = std::slice::from_raw_parts_mut((*ptr).data[2], (v_stride * height / 2) as usize);
        (y, u, v)
    };

    let mut image = YuvPlanarImageMut {
        y_plane: BufferStoreMut::Borrowed(y_plane),
        y_stride,
        u_plane: BufferStoreMut::Borrowed(u_plane),
        u_stride,
        v_plane: BufferStoreMut::Borrowed(v_plane),
        v_stride,
        width,
        height,
    };

    rgba_to_yuv420(
        &mut image,
        input_frame.data(0),
        width * 4,
        YuvRange::Limited,
        YuvStandardMatrix::Bt601,
        YuvConversionMode::Fast,
    );
}

#[test]
fn test_bench_ffmpeg_alloc() {
    const S: usize = 1000_0;
    let mut v = Vec::with_capacity(S);
    let now = Instant::now();
    for i in 0..S {
        let mut input_frame = ffmpeg_next::util::frame::video::Video::empty();
        unsafe {
            input_frame.alloc(pixel::Pixel::RGBA, 3840, 2160);
        }
        v.push(input_frame.data(0)[0] as usize);
    }

    println!("{:?}ms", now.elapsed().as_millis());

    let mut sum: usize = 0;
    for i in 0..v.len() {
        sum.wrapping_add(v[i]);
    }

    println!("Sum: {}", sum);
}

#[test]
fn test_bench_frame_memcpy() {
    const S: usize = 1000;
    let mut input_frame = ffmpeg_next::util::frame::video::Video::empty();
    unsafe {
        input_frame.alloc(pixel::Pixel::RGBA, 1920, 1080);
    }
    let mut output_frame = ffmpeg_next::util::frame::video::Video::empty();
    unsafe {
        output_frame.alloc(pixel::Pixel::RGBA, 1920, 1080);
    }
    let now = Instant::now();
    for i in 0..S {
        unsafe {
            let mut data_in = input_frame.data_mut(0);
            let mut data_out = output_frame.data_mut(0);

            data_out.copy_from_slice(data_in);
        }
    }

    println!("{:?}ms", now.elapsed().as_millis());
}

#[test]
fn bench_encode_video() {
    let width = 1920;
    let height = 1080;
    const S: usize = 100;
    // video codec settings
    let v_codec =
        ffmpeg_next::encoder::find(ffmpeg_next::codec::Id::H264).expect("H.264 encoder not found");

    let mut v_enc_ctx = ffmpeg_next::codec::context::Context::new_with_codec(v_codec);
    let mut v_enc = v_enc_ctx.encoder().video().unwrap();
    v_enc.set_width(width);
    v_enc.set_height(height);
    v_enc.set_format(Pixel::YUV420P);
    v_enc.set_time_base((1, 30));
    v_enc.set_gop(10);

    let mut v_opts = Dictionary::new();

    v_opts.set("preset", "ultrafast");
    v_opts.set("crf", "23");
    v_opts.set("tune", "zerolatency");

    let mut v_enc = v_enc
        .open_as_with(v_codec, v_opts)
        .expect("Failed to open encoder");
    let mut input_frame = ffmpeg_next::util::frame::video::Video::empty();
    unsafe {
        input_frame.alloc(pixel::Pixel::RGBA, width, height);
    }

    let mut output_frame = ffmpeg_next::util::frame::video::Video::empty();
    unsafe {
        output_frame.alloc(Pixel::YUV420P, width, height);
    }
    let mut packet = ffmpeg_next::Packet::empty();
    let mut i = 0;
    let now = Instant::now();
    for _ in 0..S {
        output_frame.set_pts(Some(i as i64));
        v_enc.send_frame(&output_frame);
        loop {
            let mut packet = ffmpeg_next::Packet::empty();
            match v_enc.receive_packet(&mut packet) {
                Ok(_) => {}
                Err(e) => {
                    break;
                } // EAGAIN or EOF
            }
        }
        i += 1;
    }
    v_enc.send_eof();
    loop {
        let mut packet = ffmpeg_next::Packet::empty();
        match v_enc.receive_packet(&mut packet) {
            Ok(_) => {}
            Err(e) => {
                break;
            } // EAGAIN or EOF
        }
    }

    let elapsed = now.elapsed();
    println!("Encoding time: {:?}", elapsed);
}

#[test]
fn test_bench_scaler() {
    let width = 1920;
    let height = 1080;
    let mut scaler = scaling::context::Context::get(
        Pixel::RGBA,
        width,
        height,
        Pixel::YUV420P,
        width,
        height,
        scaling::flag::Flags::POINT,
    )
    .unwrap();
    const S: usize = 1000;
    let mut input_frame = ffmpeg_next::util::frame::video::Video::empty();
    unsafe {
        input_frame.alloc(pixel::Pixel::RGBA, 1920, 1080);
    }
    let mut output_frame = ffmpeg_next::util::frame::video::Video::empty();
    unsafe {
        output_frame.alloc(pixel::Pixel::YUV420P, 1920, 1080);
    }

    let now = Instant::now();

    use yuv::rgba_to_yuv420;
    use yuv::BufferStoreMut;
    use yuv::YuvConversionMode;
    use yuv::YuvPlanarImageMut;
    use yuv::YuvRange;
    use yuv::YuvStandardMatrix;

    let y_stride = output_frame.plane_width(0);
    let u_stride = output_frame.plane_width(1);
    let v_stride = output_frame.plane_width(2);
    let width = output_frame.width();
    let height = output_frame.height();

    let (y_plane, u_plane, v_plane) = unsafe {
        let ptr = output_frame.as_mut_ptr();
        let y = std::slice::from_raw_parts_mut((*ptr).data[0], (y_stride * height) as usize);
        let u = std::slice::from_raw_parts_mut((*ptr).data[1], (u_stride * height / 2) as usize);
        let v = std::slice::from_raw_parts_mut((*ptr).data[2], (v_stride * height / 2) as usize);
        (y, u, v)
    };

    let mut image = YuvPlanarImageMut {
        y_plane: BufferStoreMut::Borrowed(y_plane),
        y_stride,
        u_plane: BufferStoreMut::Borrowed(u_plane),
        u_stride,
        v_plane: BufferStoreMut::Borrowed(v_plane),
        v_stride,
        width,
        height,
    };

    for i in 0..S {
        scaler.run(&input_frame, &mut output_frame).unwrap();
        // rgba_to_yuv420(
        //     &mut image,
        //     input_frame.data(0),
        //     width * 4,
        //     YuvRange::Limited,
        //     YuvStandardMatrix::Bt601,
        //     YuvConversionMode::Balanced,
        // );
    }
    println!("{:?}ms", now.elapsed().as_millis());
}
