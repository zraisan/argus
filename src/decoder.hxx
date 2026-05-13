#pragma once

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}

struct Decoder {
  AVFormatContext *formatCtx = nullptr;
  AVCodecContext *codecCtx = nullptr;
  AVFrame *frame = nullptr;
  AVPacket *packet = nullptr;
  int videoStreamIndex = -1;
};

bool openStream(Decoder &d, const char *url, const char *rtsp_transport);
bool nextFrame(Decoder &d);
void closeStream(Decoder &d);
