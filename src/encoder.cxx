#include "encoder.hxx"
#include "server.hxx"

#include <iostream>

extern "C" {
#include <libavutil/error.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
}

void print_av_error(const char *where, int err) {
  char buf[AV_ERROR_MAX_STRING_SIZE] = {};
  av_strerror(err, buf, sizeof(buf));
  std::cout << where << ": " << buf << std::endl;
}

bool supports_pixel_format(const AVCodec *codec, AVPixelFormat pix_fmt) {
  const void *configs = nullptr;
  int num_configs = 0;
  int ret = avcodec_get_supported_config(
      nullptr, codec, AV_CODEC_CONFIG_PIX_FORMAT, 0, &configs, &num_configs);

  if (ret < 0)
    return false;

  if (!configs)
    return true;

  const AVPixelFormat *pix_fmts = (const AVPixelFormat *)configs;
  for (int i = 0; i < num_configs; i++) {
    if (pix_fmts[i] == pix_fmt)
      return true;
  }
  return false;
}

AVRational stream_fps(AVStream *stream) {
  if (stream && stream->avg_frame_rate.num > 0 && stream->avg_frame_rate.den > 0)
    return stream->avg_frame_rate;

  if (stream && stream->r_frame_rate.num > 0 && stream->r_frame_rate.den > 0)
    return stream->r_frame_rate;

  return AVRational{30, 1};
}

bool open_encoder(Encoder &e, AVStream *in_stream,
                  AVCodecContext *decoder_codec_ctx) {
  if (!decoder_codec_ctx) {
    std::cout << "ERROR decoder codec context is null" << std::endl;
    return false;
  }

  const AVCodec *codec = avcodec_find_encoder_by_name("h264_nvenc");
  if (!codec) {
    std::cout << "ERROR could not find NVIDIA H.264 encoder h264_nvenc"
              << std::endl;
    return false;
  }

  AVPixelFormat pix_fmt = (AVPixelFormat)decoder_codec_ctx->pix_fmt;
  if (!supports_pixel_format(codec, pix_fmt)) {
    std::cout << "ERROR encoder does not support input pixel format "
              << av_get_pix_fmt_name(pix_fmt)
              << ". Convert frames before encoding." << std::endl;
    return false;
  }

  e.codec_ctx = avcodec_alloc_context3(codec);
  if (!e.codec_ctx) {
    std::cout << "ERROR could not allocate encoder context" << std::endl;
    return false;
  }

  AVRational fps = stream_fps(in_stream);
  int fps_rounded = (int)(av_q2d(fps) + 0.5);
  if (fps_rounded <= 0)
    fps_rounded = 30;

  e.codec_ctx->width = decoder_codec_ctx->width;
  e.codec_ctx->height = decoder_codec_ctx->height;
  e.codec_ctx->pix_fmt = pix_fmt;
  e.codec_ctx->time_base = av_inv_q(fps);
  e.codec_ctx->framerate = fps;
  e.codec_ctx->bit_rate = 4000000;
  e.codec_ctx->gop_size = fps_rounded;
  e.codec_ctx->max_b_frames = 0;

  AVDictionary *opts = nullptr;
  av_dict_set(&opts, "preset", "p1", 0);
  av_dict_set(&opts, "tune", "ull", 0);
  av_dict_set(&opts, "rc", "cbr", 0);

  int ret = avcodec_open2(e.codec_ctx, codec, &opts);
  av_dict_free(&opts);
  if (ret < 0) {
    print_av_error("ERROR opening encoder", ret);
    close_encoder(e);
    return false;
  }

  e.packet = av_packet_alloc();
  if (!e.packet) {
    std::cout << "ERROR could not allocate encoder packet" << std::endl;
    close_encoder(e);
    return false;
  }

  std::cout << "Encoder opened: " << codec->name << " "
            << e.codec_ctx->width << "x" << e.codec_ctx->height << " fps "
            << fps.num << "/" << fps.den << std::endl;
  return true;
}

bool serve_stream(Encoder &e, Server &s, AVFrame *frame) {
  if (!e.codec_ctx || !e.packet || !frame)
    return false;

  frame->pts = e.next_pts++;

  int ret = avcodec_send_frame(e.codec_ctx, frame);
  if (ret < 0) {
    print_av_error("ERROR sending frame to encoder", ret);
    return false;
  }

  while (true) {
    ret = avcodec_receive_packet(e.codec_ctx, e.packet);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
      return true;
    if (ret < 0) {
      print_av_error("ERROR receiving packet from encoder", ret);
      return false;
    }

    if (!write_packet(s, e.packet, e.codec_ctx->time_base))
      return false;
  }
}

bool flush_encoder(Encoder &e, Server &s) {
  if (!e.codec_ctx || !e.packet)
    return true;

  int ret = avcodec_send_frame(e.codec_ctx, nullptr);
  if (ret < 0) {
    print_av_error("ERROR sending flush to encoder", ret);
    return false;
  }

  while (true) {
    ret = avcodec_receive_packet(e.codec_ctx, e.packet);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
      return true;
    if (ret < 0) {
      print_av_error("ERROR flushing encoder", ret);
      return false;
    }

    if (!write_packet(s, e.packet, e.codec_ctx->time_base))
      return false;
  }
}

void close_encoder(Encoder &e) {
  if (e.packet)
    av_packet_free(&e.packet);
  if (e.codec_ctx)
    avcodec_free_context(&e.codec_ctx);
  e.next_pts = 0;
}
