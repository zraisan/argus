#include "postprocess.hxx"
#include <vector>

std::vector<Detection> postprocess(const float *output, // [1, max_dets, 6]
                                   int maxDets, int srcW, int srcH) {
  std::vector<Detection> dets;
  for (int i = 0; i < maxDets; i++) {
    float x1 = output[i * 6] * srcW / 640.0f;
    float y1 = output[i * 6 + 1] * srcH / 640.0f;
    float x2 = output[i * 6 + 2] * srcW / 640.0f;
    float y2 = output[i * 6 + 3] * srcH / 640.0f;
    float conf = output[i * 6 + 4];
    int classId = output[i * 6 + 5];
    if (conf <= 0.0f)
      break; // Ouput sometimes add padded 0s to the maxDets
    dets.push_back({x1, y1, x2, y2, conf, classId});
  }
  return dets;
}
