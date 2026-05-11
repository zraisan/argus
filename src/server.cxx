#include "server.hxx"
#include "ffmpeg_utils.hxx"
#include "logger.hxx"

#include <cerrno>
#include <string>

extern "C" {
#include <libavutil/error.h>
#include <libavutil/opt.h>
}

bool open_server(Server &s, const char *url, AVCodecContext *encoder_codec_ctx) {
  if (!url || !encoder_codec_ctx) {
    log_msg(LOG_ERROR, "rtsp", "invalid server input");
    return false;
  }

  log_msg(LOG_INFO, "rtsp", std::string("creating RTSP output context for ") + url);
  int ret = avformat_alloc_output_context2(&s.format_ctx, nullptr, "rtsp", url);
  if (ret < 0 || !s.format_ctx) {
    log_av_error(LOG_ERROR, "rtsp", "creating RTSP output context failed", ret);
    return false;
  }

  log_msg(LOG_INFO, "rtsp", "creating output video stream");
  s.stream = avformat_new_stream(s.format_ctx, nullptr);
  if (!s.stream) {
    log_msg(LOG_ERROR, "rtsp", "creating output stream failed");
    close_server(s);
    return false;
  }

  s.stream->time_base = encoder_codec_ctx->time_base;

  ret = avcodec_parameters_from_context(s.stream->codecpar, encoder_codec_ctx);
  if (ret < 0) {
    log_av_error(LOG_ERROR, "rtsp", "copying encoder parameters failed", ret);
    close_server(s);
    return false;
  }

  s.stream->codecpar->codec_tag = 0;
  log_msg(LOG_INFO, "rtsp",
          "output stream index=" + std::to_string(s.stream->index) +
              " time_base=" + av_rational_string(s.stream->time_base));

  if (!(s.format_ctx->oformat->flags & AVFMT_NOFILE)) {
    log_msg(LOG_INFO, "rtsp", "opening output IO");
    ret = avio_open(&s.format_ctx->pb, url, AVIO_FLAG_WRITE);
    if (ret < 0) {
      log_av_error(LOG_ERROR, "rtsp", "opening output IO failed", ret);
      close_server(s);
      return false;
    }
    s.io_opened = true;
  }

  AVDictionary *opts = nullptr;
  av_dict_set(&opts, "rtsp_transport", "tcp", 0);

  log_msg(LOG_INFO, "rtsp", "avformat_write_header started");
  ret = avformat_write_header(s.format_ctx, &opts);
  log_msg(LOG_INFO, "rtsp", "avformat_write_header finished");
  av_dict_free(&opts);
  if (ret < 0) {
    log_av_error(LOG_ERROR, "rtsp", "writing RTSP header failed", ret);
    if (ret == AVERROR(ECONNREFUSED)) {
      log_msg(LOG_ERROR, "rtsp",
              "MediaMTX is not reachable on localhost:8554");
    }
    close_server(s);
    return false;
  }

  log_msg(LOG_INFO, "rtsp", std::string("RTSP output is live at ") + url);
  return true;
}

bool write_packet(Server &s, AVPacket *packet, AVRational encoder_time_base) {
  if (!s.format_ctx || !s.stream || !packet)
    return false;

  packet->stream_index = s.stream->index;
  av_packet_rescale_ts(packet, encoder_time_base, s.stream->time_base);

  int ret = av_interleaved_write_frame(s.format_ctx, packet);
  if (ret < 0) {
    log_av_error(LOG_ERROR, "rtsp", "writing packet failed", ret);
    av_packet_unref(packet);
    return false;
  }

  av_packet_unref(packet);
  return true;
}

void close_server(Server &s) {
  log_msg(LOG_INFO, "rtsp", "closing RTSP output");
  if (s.format_ctx)
    av_write_trailer(s.format_ctx);

  if (s.io_opened && s.format_ctx && s.format_ctx->pb) {
    avio_closep(&s.format_ctx->pb);
    s.io_opened = false;
  }

  if (s.format_ctx)
    avformat_free_context(s.format_ctx);

  s.format_ctx = nullptr;
  s.stream = nullptr;
}
