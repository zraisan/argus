#pragma once

#include "NvInfer.h"
#include <cstddef>
#include <cuda_runtime_api.h>
#include <memory>
#include <vector>

struct Engine {
  std::unique_ptr<nvinfer1::IRuntime> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  std::unique_ptr<nvinfer1::IExecutionContext> context;
  std::vector<void *> gpuBuffers;
  void *inputBuffer = nullptr;
  void *outputBuffer = nullptr;
  std::size_t inputBytes = 0;
  std::size_t outputBytes = 0;
  cudaStream_t stream = nullptr;
};

Engine buildEngine(const char *onnxPath);
void runInference(Engine &e, const float *cpuInput, float *cpuOutput);
void destroyEngine(Engine &e);
