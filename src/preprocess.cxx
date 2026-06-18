#include "preprocess.hxx"
#include <algorithm>
#include <cmath>

extern "C" {
#include <libavutil/pixfmt.h>
}

void initPreprocess(Preprocessor &p, int srcW, int srcH, int srcPixFmt) {
  const float scale =
      std::min((float)p.dstW / (float)srcW, (float)p.dstH / (float)srcH);
  const int resized_w =
      std::max(1, std::min(p.dstW, (int)std::round(srcW * scale)));
  const int resized_h =
      std::max(1, std::min(p.dstH, (int)std::round(srcH * scale)));

  p.letterbox.scale = scale;
  p.letterbox.pad_x = (p.dstW - resized_w) / 2;
  p.letterbox.pad_y = (p.dstH - resized_h) / 2;
  p.letterbox.resized_w = resized_w;
  p.letterbox.resized_h = resized_h;

  p.sws = sws_getContext(srcW, srcH, (AVPixelFormat)srcPixFmt, resized_w,
                         resized_h, AV_PIX_FMT_RGB24, SWS_BILINEAR, nullptr,
                         nullptr, nullptr);
  p.rgbBuf.reset(new uint8_t[p.dstW * p.dstH * 3]);
}

std::unique_ptr<float[]> preprocess(Preprocessor &p, AVFrame *src, float *dst) {
  std::fill(p.rgbBuf.get(), p.rgbBuf.get() + p.dstW * p.dstH * 3, 114);

  uint8_t *dstRGB[1] = {p.rgbBuf.get() +
                        p.letterbox.pad_y * p.dstW * 3 +
                        p.letterbox.pad_x * 3};
  int dstStride[1] = {p.dstW * 3};
  sws_scale(p.sws, src->data, src->linesize, 0, src->height, dstRGB, dstStride);

  const int nPixels = p.dstW * p.dstH;
  const uint8_t *rgb = p.rgbBuf.get();
  for (int k = 0; k < nPixels; k++) {
    dst[k] = rgb[3 * k] / 255.0f;
    dst[nPixels + k] = rgb[3 * k + 1] / 255.0f;
    dst[2 * nPixels + k] = rgb[3 * k + 2] / 255.0f;
  }
  std::unique_ptr<float[]> frame_rgb{new float[3 * nPixels]};
  std::copy(dst, dst + 3 * nPixels, frame_rgb.get());

  return frame_rgb;
}

void destroyPreprocess(Preprocessor &p) {
  if (p.sws) {
    sws_freeContext(p.sws);
    p.sws = nullptr;
  }
}
