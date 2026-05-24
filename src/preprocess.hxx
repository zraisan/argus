#pragma once

extern "C" {
#include <libavutil/frame.h>
#include <libswscale/swscale.h>
}

#include <cstdint>
#include <memory>

struct Preprocessor {
  SwsContext *sws = nullptr;
  std::unique_ptr<uint8_t[]> rgbBuf;
  int dstW = 640;
  int dstH = 640;
};

void initPreprocess(Preprocessor &p, int srcW, int srcH, int srcPixFmt);
std::unique_ptr<float[]> preprocess(Preprocessor &p, AVFrame *src, float *dst);
void destroyPreprocess(Preprocessor &p);
