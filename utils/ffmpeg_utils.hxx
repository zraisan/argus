#pragma once

#include <string>

extern "C" {
#include <libavutil/pixfmt.h>
#include <libavutil/rational.h>
}

std::string av_rational_string(AVRational rational);
std::string av_pixel_format_name(AVPixelFormat pixel_format);
