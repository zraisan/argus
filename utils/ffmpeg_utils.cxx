#include "ffmpeg_utils.hxx"

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
