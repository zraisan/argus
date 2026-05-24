#pragma once

#include "NvInfer.h"
#include <cstddef>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>
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
  std::string inputName;
  std::string outputName;
  std::size_t inputElementsPerFrame = 0;
  std::size_t outputElementsPerFrame = 0;
  cudaStream_t stream = nullptr;
};

bool build_engine_file(const char *onnx_path, const char *engine_path);
Engine load_engine(const char *engine_path, int batch_size);
std::unique_ptr<float[]> run_inference(Engine &e, const float *cpu_input,
                                       float *cpu_output, int batch_size);
void destroy_engine(Engine &e);
