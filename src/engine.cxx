#include "engine.hxx"
#include "NvOnnxParser.h"
#include "cuda.hxx"
#include <cstdint>
#include <iostream>

using namespace nvinfer1;
using namespace nvonnxparser;

namespace {
class TRTLogger : public ILogger {
  void log(Severity severity, const char *msg) noexcept override {
    if (severity <= Severity::kWARNING)
      std::cout << msg << std::endl;
  }
} logger;
} // namespace

Engine buildEngine(const char *onnxPath) {
  Engine e;

  std::unique_ptr<IBuilder> builder{createInferBuilder(logger)};
  std::unique_ptr<INetworkDefinition> network{builder->createNetworkV2(
      1U << static_cast<uint32_t>(
          NetworkDefinitionCreationFlag::kSTRONGLY_TYPED))};
  std::unique_ptr<IParser> parser{createParser(*network, logger)};

  parser->parseFromFile(onnxPath,
                        static_cast<int32_t>(ILogger::Severity::kWARNING));
  for (int32_t i = 0; i < parser->getNbErrors(); ++i)
    std::cout << parser->getError(i)->desc() << std::endl;

  std::unique_ptr<IBuilderConfig> config{builder->createBuilderConfig()};
  IOptimizationProfile *profile = builder->createOptimizationProfile();

  for (int32_t i = 0; i < network->getNbInputs(); i++) {
    ITensor *input = network->getInput(i);
    Dims inputDims = input->getDimensions();
    Dims minDims = inputDims, optDims = inputDims, maxDims = inputDims;

    minDims.d[0] = 1;
    optDims.d[0] = 4;
    maxDims.d[0] = 16;
    if (inputDims.nbDims >= 4) {
      minDims.d[2] = optDims.d[2] = maxDims.d[2] = 640;
      minDims.d[3] = optDims.d[3] = maxDims.d[3] = 640;
    }

    profile->setDimensions(input->getName(), OptProfileSelector::kMIN, minDims);
    profile->setDimensions(input->getName(), OptProfileSelector::kOPT, optDims);
    profile->setDimensions(input->getName(), OptProfileSelector::kMAX, maxDims);
  }
  config->addOptimizationProfile(profile);

  std::unique_ptr<IHostMemory> serializedModel{
      builder->buildSerializedNetwork(*network, *config)};

  e.runtime.reset(createInferRuntime(logger));
  e.engine.reset(e.runtime->deserializeCudaEngine(serializedModel->data(),
                                                  serializedModel->size()));
  e.context.reset(e.engine->createExecutionContext());

  for (int32_t i = 0; i < network->getNbInputs(); i++) {
    ITensor *input = network->getInput(i);
    Dims dims = input->getDimensions();
    dims.d[0] = 1;
    if (dims.nbDims >= 4) {
      dims.d[2] = 640;
      dims.d[3] = 640;
    }
    e.context->setInputShape(input->getName(), dims);
  }

  e.gpuBuffers.assign(e.engine->getNbIOTensors(), nullptr);
  for (int32_t i = 0; i < e.engine->getNbIOTensors(); i++) {
    auto const name = e.engine->getIOTensorName(i);
    Dims dims = e.context->getTensorShape(name);
    uint64_t elements = 1;
    for (int j = 0; j < dims.nbDims; j++)
      elements *= dims.d[j];
    DataType type = e.engine->getTensorDataType(name);
    std::size_t bytes = elements * elemSize(type);
    cudaMalloc(&e.gpuBuffers[i], bytes);
    e.context->setTensorAddress(name, e.gpuBuffers[i]);

    if (e.engine->getTensorIOMode(name) == TensorIOMode::kINPUT) {
      e.inputBuffer = e.gpuBuffers[i];
      e.inputBytes = bytes;
    } else {
      e.outputBuffer = e.gpuBuffers[i];
      e.outputBytes = bytes;
    }
  }

  cudaStreamCreate(&e.stream);
  return e;
}

void runInference(Engine &e, const float *cpuInput, float *cpuOutput) {
  cudaMemcpyAsync(e.inputBuffer, cpuInput, e.inputBytes,
                  cudaMemcpyHostToDevice, e.stream);
  e.context->enqueueV3(e.stream);
  cudaMemcpyAsync(cpuOutput, e.outputBuffer, e.outputBytes,
                  cudaMemcpyDeviceToHost, e.stream);
  cudaStreamSynchronize(e.stream);
}

void destroyEngine(Engine &e) {
  for (void *buf : e.gpuBuffers)
    if (buf)
      cudaFree(buf);
  e.gpuBuffers.clear();
  if (e.stream)
    cudaStreamDestroy(e.stream);
}
