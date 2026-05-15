#include "server.hxx"
#include "logger.hxx"
#include "streams.hxx"

#include <cerrno>
#include <string>

extern "C" {
#include <libavutil/error.h>
#include <libavutil/opt.h>
}

bool open_server(Server &s, const char *url,
                 AVCodecContext *encoder_codec_ctx) {
  if (!url || !encoder_codec_ctx) {
    log_msg(LOG_ERROR, "server", "invalid server input");
    return false;
  }

  StreamType type = stream_type_from_url(url);

  const char *muxer_name = nullptr;
  const char *output_type = nullptr;
  if (type == StreamType::Rtsp)
    muxer_name = output_type = "rtsp";
  else if (type == StreamType::Hls)
    muxer_name = output_type = "hls";
  else if (type == StreamType::Mp4)
    muxer_name = output_type = "mp4";
  else if (type == StreamType::MpegTs) {
    muxer_name = "mpegts";
    output_type = "mpeg-ts";
  } else if (type == StreamType::Matroska)
    muxer_name = output_type = "matroska";
  else if (type == StreamType::Webrtc)
    muxer_name = output_type = "whip";
  else {
    log_msg(LOG_ERROR, "server",
            "could not determine output type from URL. Use rtsp://, .m3u8, "
            ".mp4, .ts, .mkv, or .webm");
    return false;
  }

  log_msg(LOG_INFO, "server",
          std::string("creating ") + output_type + " output context for " +
              url);
  int ret =
      avformat_alloc_output_context2(&s.format_ctx, nullptr, muxer_name, url);

  if (ret < 0 || !s.format_ctx) {
    log_av_error(LOG_ERROR, "server", "creating output context failed", ret);
    return false;
  }

  log_msg(LOG_INFO, "server", "creating output video stream");
  s.stream = avformat_new_stream(s.format_ctx, nullptr);
  if (!s.stream) {
    log_msg(LOG_ERROR, "server", "creating output stream failed");
    close_server(s);
    return false;
  }

  s.stream->time_base = encoder_codec_ctx->time_base;

  ret = avcodec_parameters_from_context(s.stream->codecpar, encoder_codec_ctx);
  if (ret < 0) {
    log_av_error(LOG_ERROR, "server", "copying encoder parameters failed", ret);
    close_server(s);
    return false;
  }

  s.stream->codecpar->codec_tag = 0;
  log_msg(LOG_INFO, "server",
          "output stream index=" + std::to_string(s.stream->index) +
              " time_base=" + av_rational_string(s.stream->time_base));

  if (!(s.format_ctx->oformat->flags & AVFMT_NOFILE)) {
    log_msg(LOG_INFO, "server", "opening output IO");
    ret = avio_open(&s.format_ctx->pb, url, AVIO_FLAG_WRITE);
    if (ret < 0) {
      log_av_error(LOG_ERROR, "server", "opening output IO failed", ret);
      close_server(s);
      return false;
    }
    s.io_opened = true;
  }

  AVDictionary *opts = nullptr;
  if (type == StreamType::Rtsp)
    av_dict_set(&opts, "rtsp_transport", "tcp", 0);
  else if (type == StreamType::Hls) {
    std::string segment_filename = std::string(url);
    const std::size_t ext_pos = segment_filename.rfind(".m3u8");
    if (ext_pos != std::string::npos)
      segment_filename.replace(ext_pos, 5, "_%03d.ts");

    av_dict_set(&opts, "hls_time", "2", 0);
    av_dict_set(&opts, "hls_list_size", "5", 0);
    av_dict_set(&opts, "hls_flags", "delete_segments", 0);
    av_dict_set(&opts, "hls_delete_threshold", "2", 0);
    av_dict_set(&opts, "hls_segment_filename", segment_filename.c_str(), 0);
  }

  log_msg(LOG_INFO, "server", "avformat_write_header started");
  ret = avformat_write_header(s.format_ctx, &opts);
  log_msg(LOG_INFO, "server", "avformat_write_header finished");
  av_dict_free(&opts);
  if (ret < 0) {
    log_av_error(LOG_ERROR, "server", "writing output header failed", ret);
    if (type == StreamType::Rtsp && ret == AVERROR(ECONNREFUSED))
      log_msg(LOG_ERROR, "server", "MediaMTX is not reachable on localhost:8554");
    close_server(s);
    return false;
  }

  log_msg(LOG_INFO, "server", std::string("output is ready at ") + url);
  return true;
}

bool write_packet(Server &s, AVPacket *packet, AVRational encoder_time_base) {
  if (!s.format_ctx || !s.stream || !packet)
    return false;

  packet->stream_index = s.stream->index;
  av_packet_rescale_ts(packet, encoder_time_base, s.stream->time_base);

  int ret = av_interleaved_write_frame(s.format_ctx, packet);
  if (ret < 0) {
    log_av_error(LOG_ERROR, "server", "writing packet failed", ret);
    av_packet_unref(packet);
    return false;
  }

  av_packet_unref(packet);
  return true;
}

void close_server(Server &s) {
  log_msg(LOG_INFO, "server", "closing output");
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
