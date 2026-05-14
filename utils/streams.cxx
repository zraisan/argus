#include "streams.hxx"

#include <algorithm>
#include <cctype>
#include <sstream>

extern "C" {
#include <libavutil/pixdesc.h>
}

std::string av_rational_string(AVRational rational) {
  std::ostringstream out;
  out << rational.num << "/" << rational.den;
  return out.str();
}

std::string av_pixel_format_name(AVPixelFormat pixel_format) {
  const char *name = av_get_pix_fmt_name(pixel_format);
  return name ? name : "(unknown)";
}

StreamType stream_type_from_url(const std::string &url) {
  std::string value = url;
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return (char)std::tolower(c); });

  const std::size_t query_pos = value.find_first_of("?#");
  const std::string path = value.substr(0, query_pos);

  if (path.rfind("rtsp://", 0) == 0 || path.rfind("rtsps://", 0) == 0)
    return StreamType::Rtsp;

  if (path.rfind("webrtc://", 0) == 0 || path.rfind("whep://", 0) == 0 ||
      path.rfind("wheps://", 0) == 0 || path.rfind("whip://", 0) == 0 ||
      path.rfind("whips://", 0) == 0)
    return StreamType::Webrtc;

  if (path.rfind("hls://", 0) == 0 ||
      (path.size() >= 5 && path.compare(path.size() - 5, 5, ".m3u8") == 0))
    return StreamType::Hls;

  if (path.size() >= 4 && path.compare(path.size() - 4, 4, ".mp4") == 0)
    return StreamType::Mp4;

  if (path.size() >= 3 && path.compare(path.size() - 3, 3, ".ts") == 0)
    return StreamType::MpegTs;

  if (path.size() >= 4 && path.compare(path.size() - 4, 4, ".mkv") == 0)
    return StreamType::Matroska;

  if (path.size() >= 5 && path.compare(path.size() - 5, 5, ".webm") == 0)
    return StreamType::Matroska;

  if (path.rfind("http://", 0) == 0 || path.rfind("https://", 0) == 0)
    return StreamType::Http;

  if (path.rfind("file://", 0) == 0 || path.find("://") == std::string::npos)
    return StreamType::File;

  return StreamType::Unknown;
}
