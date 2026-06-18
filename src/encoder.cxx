#include "encoder.hxx"
#include "logger.hxx"
#include "server.hxx"
#include "streams.hxx"

#include <string>

extern "C" {
#include <libavutil/error.h>
#include <libavutil/opt.h>
}

namespace {

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

bool sane_fps(AVRational fps) {
  if (fps.num <= 0 || fps.den <= 0)
    return false;

  double value = av_q2d(fps);
  return value >= 1.0 && value <= 120.0;
}

AVRational stream_fps(AVStream *stream) {
  if (stream && sane_fps(stream->avg_frame_rate))
    return stream->avg_frame_rate;

  if (stream && sane_fps(stream->r_frame_rate))
    return stream->r_frame_rate;

  if (stream) {
    log_msg(LOG_WARNING, "encoder",
            "using fallback fps 30/1, avg_frame_rate=" +
                av_rational_string(stream->avg_frame_rate) +
                " r_frame_rate=" + av_rational_string(stream->r_frame_rate));
  }
  return AVRational{30, 1};
}

} // namespace

bool open_encoder(Encoder &e, AVStream *in_stream,
                  AVCodecContext *decoder_codec_ctx) {
  if (!decoder_codec_ctx) {
    log_msg(LOG_ERROR, "encoder", "decoder codec context is null");
    return false;
  }

  return open_encoder(e, decoder_codec_ctx->width, decoder_codec_ctx->height,
                      (AVPixelFormat)decoder_codec_ctx->pix_fmt,
                      stream_fps(in_stream));
}

bool open_encoder(Encoder &e, int width, int height, AVPixelFormat pix_fmt,
                  AVRational fps) {
  log_msg(LOG_INFO, "encoder", "finding NVIDIA H.264 encoder h264_nvenc");
  const AVCodec *codec = avcodec_find_encoder_by_name("h264_nvenc");
  if (!codec) {
    log_msg(LOG_ERROR, "encoder",
            "could not find NVIDIA H.264 encoder h264_nvenc");
    return false;
  }

  if (!supports_pixel_format(codec, pix_fmt)) {
    log_msg(LOG_ERROR, "encoder",
            "encoder does not support input pixel format " +
                av_pixel_format_name(pix_fmt) +
                ". Convert frames before encoding.");
    return false;
  }

  log_msg(LOG_INFO, "encoder",
          std::string("allocating encoder context for ") + codec->name);
  e.codec_ctx = avcodec_alloc_context3(codec);
  if (!e.codec_ctx) {
    log_msg(LOG_ERROR, "encoder", "could not allocate encoder context");
    return false;
  }

  if (!sane_fps(fps)) {
    log_msg(LOG_WARNING, "encoder", "using fallback fps 30/1");
    fps = AVRational{30, 1};
  }

  int fps_rounded = (int)(av_q2d(fps) + 0.5);
  if (fps_rounded <= 0)
    fps_rounded = 30;

  e.codec_ctx->width = width;
  e.codec_ctx->height = height;
  e.codec_ctx->pix_fmt = pix_fmt;
  e.codec_ctx->time_base = av_inv_q(fps);
  e.codec_ctx->framerate = fps;
  e.codec_ctx->bit_rate = 4000000;
  e.codec_ctx->gop_size = fps_rounded;
  e.codec_ctx->max_b_frames = 0;

  log_msg(LOG_INFO, "encoder",
          "config width=" + std::to_string(e.codec_ctx->width) +
              " height=" + std::to_string(e.codec_ctx->height) +
              " pix_fmt=" + av_pixel_format_name(e.codec_ctx->pix_fmt) +
              " fps=" + av_rational_string(fps) +
              " time_base=" + av_rational_string(e.codec_ctx->time_base) +
              " bitrate=" + std::to_string(e.codec_ctx->bit_rate) +
              " gop=" + std::to_string(e.codec_ctx->gop_size) +
              " max_b_frames=" + std::to_string(e.codec_ctx->max_b_frames));

  AVDictionary *opts = nullptr;
  av_dict_set(&opts, "preset", "p1", 0);
  av_dict_set(&opts, "tune", "ull", 0);
  av_dict_set(&opts, "rc", "cbr", 0);
  log_msg(LOG_INFO, "encoder", "NVENC options preset=p1 tune=ull rc=cbr");

  int ret = 0;
  log_msg(LOG_INFO, "encoder", "avcodec_open2 started");
  ret = avcodec_open2(e.codec_ctx, codec, &opts);
  log_msg(LOG_INFO, "encoder", "avcodec_open2 finished");
  av_dict_free(&opts);
  if (ret < 0) {
    log_av_error(LOG_ERROR, "encoder", "opening encoder failed", ret);
    close_encoder(e);
    return false;
  }

  e.packet = av_packet_alloc();
  if (!e.packet) {
    log_msg(LOG_ERROR, "encoder", "could not allocate encoder packet");
    close_encoder(e);
    return false;
  }

  log_msg(LOG_INFO, "encoder", std::string("encoder opened: ") + codec->name);
  return true;
}

bool serve_stream(Encoder &e, Server &s, AVFrame *frame) {
  if (!e.codec_ctx || !e.packet || !frame)
    return false;

  frame->pts = e.next_pts++;

  int ret = avcodec_send_frame(e.codec_ctx, frame);
  if (ret < 0) {
    log_av_error(LOG_ERROR, "encoder", "sending frame to encoder failed", ret);
    return false;
  }

  while (true) {
    ret = avcodec_receive_packet(e.codec_ctx, e.packet);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
      return true;
    if (ret < 0) {
      log_av_error(LOG_ERROR, "encoder", "receiving packet from encoder failed",
                   ret);
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
    log_av_error(LOG_ERROR, "encoder", "sending flush to encoder failed", ret);
    return false;
  }

  while (true) {
    ret = avcodec_receive_packet(e.codec_ctx, e.packet);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
      return true;
    if (ret < 0) {
      log_av_error(LOG_ERROR, "encoder", "flushing encoder failed", ret);
      return false;
    }

    if (!write_packet(s, e.packet, e.codec_ctx->time_base))
      return false;
  }
}

void close_encoder(Encoder &e) {
  log_msg(LOG_INFO, "encoder", "closing encoder");
  if (e.packet)
    av_packet_free(&e.packet);
  if (e.codec_ctx)
    avcodec_free_context(&e.codec_ctx);
  e.next_pts = 0;
}
