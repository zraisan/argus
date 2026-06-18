#include "postprocess.hxx"
#include <algorithm>
#include <cstdint>
#include <vector>

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/pixfmt.h>
}

namespace {

struct YuvColor {
  uint8_t y, u, v;
};

namespace colors {
constexpr YuvColor White = {235, 128, 128};
constexpr YuvColor Red = {76, 84, 255};
constexpr YuvColor Green = {149, 43, 21};
constexpr YuvColor Blue = {29, 255, 107};
constexpr YuvColor Yellow = {210, 16, 146};
constexpr YuvColor Cyan = {170, 166, 16};
constexpr YuvColor Magenta = {106, 202, 222};
constexpr YuvColor Orange = {156, 47, 200};
constexpr YuvColor Purple = {69, 192, 168};
constexpr YuvColor Pink = {180, 122, 191};
constexpr YuvColor Lime = {184, 41, 87};
constexpr YuvColor Teal = {120, 149, 84};
} // namespace colors

} // namespace

std::vector<Detection> postprocess(const float *output, // [1, max_dets, 6]
                                   int maxDets, int srcW, int srcH,
                                   const LetterboxTransform &letterbox) {
  std::vector<Detection> dets;
  for (int i = 0; i < maxDets; i++) {
    float x1 = (output[i * 6] - letterbox.pad_x) / letterbox.scale;
    float y1 = (output[i * 6 + 1] - letterbox.pad_y) / letterbox.scale;
    float x2 = (output[i * 6 + 2] - letterbox.pad_x) / letterbox.scale;
    float y2 = (output[i * 6 + 3] - letterbox.pad_y) / letterbox.scale;
    float conf = output[i * 6 + 4];
    int classId = output[i * 6 + 5];
    if (conf <= 0.0f)
      break; // Ouput sometimes add padded 0s to the maxDets
    x1 = std::clamp(x1, 0.0f, (float)(srcW - 1));
    y1 = std::clamp(y1, 0.0f, (float)(srcH - 1));
    x2 = std::clamp(x2, 0.0f, (float)(srcW - 1));
    y2 = std::clamp(y2, 0.0f, (float)(srcH - 1));
    dets.push_back({x1, y1, x2, y2, conf, classId});
  }
  return dets;
}

namespace {

const YuvColor palette[] = {
    colors::White, colors::Red,    colors::Green, colors::Blue,
    colors::Yellow, colors::Cyan,  colors::Magenta, colors::Orange,
    colors::Purple, colors::Pink,  colors::Lime,    colors::Teal,
};
constexpr int kPaletteSize = sizeof(palette) / sizeof(palette[0]);

const YuvColor &colorForClass(int classId) {
  return palette[classId % kPaletteSize];
}

void setYuv420pPixel(AVFrame *frame, int x, int y, const YuvColor &c) {
  frame->data[0][y * frame->linesize[0] + x] = c.y;
  frame->data[1][(y >> 1) * frame->linesize[1] + (x >> 1)] = c.u;
  frame->data[2][(y >> 1) * frame->linesize[2] + (x >> 1)] = c.v;
}

void setNv12Pixel(AVFrame *frame, int x, int y, const YuvColor &c) {
  frame->data[0][y * frame->linesize[0] + x] = c.y;

  uint8_t *uv = frame->data[1] + (y >> 1) * frame->linesize[1] + (x & ~1);
  uv[0] = c.u;
  uv[1] = c.v;
}

template <typename SetPixel>
void drawBox(AVFrame *frame, int x1, int y1, int x2, int y2,
             const YuvColor &color, SetPixel set_pixel, int thickness = 2) {
  for (int t = 0; t < thickness; t++) {
    for (int x = x1; x <= x2; x++) {
      set_pixel(frame, x, y1 + t, color); // top
      set_pixel(frame, x, y2 - t, color); // bottom
    }
    for (int y = y1; y <= y2; y++) {
      set_pixel(frame, x1 + t, y, color); // left
      set_pixel(frame, x2 - t, y, color); // right
    }
  }
}

} // namespace

void draw_box(const std::vector<Detection> &dets, AVFrame *frame) {
  if (av_frame_make_writable(frame) < 0)
    return;

  int width = frame->width;
  int height = frame->height;
  for (const auto &d : dets) {
    int ix1 = std::clamp((int)d.x1, 0, width - 1);
    int iy1 = std::clamp((int)d.y1, 0, height - 1);
    int ix2 = std::clamp((int)d.x2, 0, width - 1);
    int iy2 = std::clamp((int)d.y2, 0, height - 1);
    if (ix2 <= ix1 || iy2 <= iy1)
      continue;

    const YuvColor &color = colorForClass(d.classId);
    if (frame->format == AV_PIX_FMT_NV12)
      drawBox(frame, ix1, iy1, ix2, iy2, color, setNv12Pixel);
    else if (frame->format == AV_PIX_FMT_YUV420P ||
             frame->format == AV_PIX_FMT_YUVJ420P)
      drawBox(frame, ix1, iy1, ix2, iy2, color, setYuv420pPixel);
  }
}
