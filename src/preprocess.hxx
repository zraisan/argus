#pragma once

extern "C" {
#include <libavutil/frame.h>
#include <libswscale/swscale.h>
}

#include <cstdint>
#include <memory>

struct LetterboxTransform {
  float scale = 1.0f;
  int pad_x = 0;
  int pad_y = 0;
  int resized_w = 640;
  int resized_h = 640;
};

struct Preprocessor {
  SwsContext *sws = nullptr;
  std::unique_ptr<uint8_t[]> rgbBuf;
  int dstW = 640;
  int dstH = 640;
  LetterboxTransform letterbox;
};

void initPreprocess(Preprocessor &p, int srcW, int srcH, int srcPixFmt);
std::unique_ptr<float[]> preprocess(Preprocessor &p, AVFrame *src, float *dst);
void destroyPreprocess(Preprocessor &p);
