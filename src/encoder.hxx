#pragma once

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/frame.h>
}

struct Server;

struct Encoder {
  AVCodecContext *codec_ctx = nullptr;
  AVPacket *packet = nullptr;
  int64_t next_pts = 0;
};

bool open_encoder(Encoder &e, AVStream *in_stream,
                  AVCodecContext *decoder_codec_ctx);
bool serve_stream(Encoder &e, Server &s, AVFrame *frame);
bool flush_encoder(Encoder &e, Server &s);
void close_encoder(Encoder &e);
