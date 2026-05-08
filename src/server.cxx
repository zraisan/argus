#include "server.hxx"

#include <iostream>

extern "C" {
#include <libavutil/error.h>
#include <libavutil/opt.h>
}

void print_server_error(const char *where, int err) {
  char buf[AV_ERROR_MAX_STRING_SIZE] = {};
  av_strerror(err, buf, sizeof(buf));
  std::cout << where << ": " << buf << std::endl;
}

bool open_server(Server &s, const char *url, AVCodecContext *encoder_codec_ctx) {
  if (!url || !encoder_codec_ctx) {
    std::cout << "ERROR invalid server input" << std::endl;
    return false;
  }

  int ret = avformat_alloc_output_context2(&s.format_ctx, nullptr, "rtsp", url);
  if (ret < 0 || !s.format_ctx) {
    print_server_error("ERROR creating RTSP output context", ret);
    return false;
  }

  s.stream = avformat_new_stream(s.format_ctx, nullptr);
  if (!s.stream) {
    std::cout << "ERROR creating output stream" << std::endl;
    close_server(s);
    return false;
  }

  s.stream->time_base = encoder_codec_ctx->time_base;

  ret = avcodec_parameters_from_context(s.stream->codecpar, encoder_codec_ctx);
  if (ret < 0) {
    print_server_error("ERROR copying encoder parameters to stream", ret);
    close_server(s);
    return false;
  }

  s.stream->codecpar->codec_tag = 0;

  if (!(s.format_ctx->oformat->flags & AVFMT_NOFILE)) {
    ret = avio_open(&s.format_ctx->pb, url, AVIO_FLAG_WRITE);
    if (ret < 0) {
      print_server_error("ERROR opening output IO", ret);
      close_server(s);
      return false;
    }
    s.io_opened = true;
  }

  AVDictionary *opts = nullptr;
  av_dict_set(&opts, "rtsp_transport", "tcp", 0);

  ret = avformat_write_header(s.format_ctx, &opts);
  av_dict_free(&opts);
  if (ret < 0) {
    print_server_error("ERROR writing RTSP header", ret);
    close_server(s);
    return false;
  }

  std::cout << "RTSP output opened: " << url << std::endl;
  return true;
}

bool write_packet(Server &s, AVPacket *packet, AVRational encoder_time_base) {
  if (!s.format_ctx || !s.stream || !packet)
    return false;

  packet->stream_index = s.stream->index;
  av_packet_rescale_ts(packet, encoder_time_base, s.stream->time_base);

  int ret = av_interleaved_write_frame(s.format_ctx, packet);
  if (ret < 0) {
    print_server_error("ERROR writing packet", ret);
    av_packet_unref(packet);
    return false;
  }

  av_packet_unref(packet);
  return true;
}

void close_server(Server &s) {
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
