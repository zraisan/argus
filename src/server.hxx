#pragma once

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}

struct Server {
  AVFormatContext *format_ctx = nullptr;
  AVStream *stream = nullptr;
  bool io_opened = false;
};

bool open_server(Server &s, const char *url, AVCodecContext *encoder_codec_ctx);
bool write_packet(Server &s, AVPacket *packet, AVRational encoder_time_base);
void close_server(Server &s);
