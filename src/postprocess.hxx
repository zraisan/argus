#pragma once

#include <vector>

struct Detection {
  float x1, y1, x2, y2;
  float conf;
  int classId;
};

std::vector<Detection> postprocess(const float *output, // [1, max_dets, 6]
                                   int maxDets, int srcW, int srcH);
