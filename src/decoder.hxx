#pragma once

#include "postprocess.hxx"
#include "streams.hxx"
#include <memory>
#include <utility>
#include <vector>
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/frame.h>
}

struct Decoder {
  AVFormatContext *formatCtx = nullptr;
  AVCodecContext *codecCtx = nullptr;
  AVFrame *frame = nullptr;
  AVPacket *packet = nullptr;
  int videoStreamIndex = -1;
  StreamType type = StreamType::Unknown;
};

struct StreamFrame {
  AVFrame *frame = nullptr;
  int stream_index = -1;
  int64_t frame_id = 0;
  std::unique_ptr<float[]> frame_rgb;
  std::unique_ptr<float[]> output;
  std::vector<Detection> detections;

  StreamFrame() = default;
  StreamFrame(AVFrame *frame, int stream_index, int64_t frame_id)
      : frame(frame), stream_index(stream_index), frame_id(frame_id) {}

  ~StreamFrame() {
    if (frame)
      av_frame_free(&frame);
  }

  StreamFrame(const StreamFrame &) = delete;
  StreamFrame &operator=(const StreamFrame &) = delete;

  StreamFrame(StreamFrame &&other) noexcept { move_from(std::move(other)); }

  StreamFrame &operator=(StreamFrame &&other) noexcept {
    if (this != &other) {
      if (frame)
        av_frame_free(&frame);
      move_from(std::move(other));
    }
    return *this;
  }

private:
  void move_from(StreamFrame &&other) noexcept {
    frame = other.frame;
    other.frame = nullptr;
    stream_index = other.stream_index;
    frame_id = other.frame_id;
    frame_rgb = std::move(other.frame_rgb);
    output = std::move(other.output);
    detections = std::move(other.detections);
  }
};

bool openStream(Decoder &d, const char *url);
bool nextFrame(Decoder &d);
void closeStream(Decoder &d);
