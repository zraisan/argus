#pragma once

#include "NvInferRuntime.h"
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

void preprocess(const uint8_t *src, float *dst);

void runLiveStream(nvinfer1::IExecutionContext *trtContext,
                   void *inputBuffer, std::size_t inputBytes,
                   void *outputBuffer, std::size_t outputBytes,
                   cudaStream_t stream);
