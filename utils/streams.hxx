#pragma once

#include <string>

extern "C" {
#include <libavutil/pixfmt.h>
#include <libavutil/rational.h>
}

enum class StreamType {
  Unknown,
  File,
  Http,
  Hls,
  Mp4,
  MpegTs,
  Matroska,
  Rtsp,
  Webrtc
};

std::string av_rational_string(AVRational rational);
std::string av_pixel_format_name(AVPixelFormat pixel_format);
StreamType stream_type_from_url(const std::string &url);
