#include "preprocess.hxx"
#include "decoder.hxx"
#include <algorithm>
#include <utility>

extern "C" {
#include <libavutil/pixfmt.h>
}

void initPreprocess(Preprocessor &p, int srcW, int srcH, int srcPixFmt) {
  p.sws =
      sws_getContext(srcW, srcH, (AVPixelFormat)srcPixFmt, p.dstW, p.dstH,
                     AV_PIX_FMT_RGB24, SWS_BILINEAR, nullptr, nullptr, nullptr);
  p.rgbBuf.reset(new uint8_t[p.dstW * p.dstH * 3]);
}

std::unique_ptr<float[]> preprocess(Preprocessor &p, AVFrame *src, float *dst) {
  uint8_t *dstRGB[1] = {p.rgbBuf.get()};
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
