#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "cuda.hxx"
#include <NvInferRuntime.h>
#include <NvInferRuntimeBase.h>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <iostream>
#include <memory>
#include <vector>

using namespace nvinfer1;
using namespace nvonnxparser;

class Logger : public ILogger {
  void log(Severity severity, const char *msg) noexcept override {
    // suppress info-level messages
    if (severity <= Severity::kWARNING)
      std::cout << msg << std::endl;
  }
} logger;

int main() {
  std::unique_ptr<IBuilder> builder{createInferBuilder(logger)};
  std::unique_ptr<INetworkDefinition> network{builder->createNetworkV2(
      1U << static_cast<uint32_t>(
          NetworkDefinitionCreationFlag::kSTRONGLY_TYPED))};
  std::unique_ptr<IParser> parser{createParser(*network, logger)};
  const char *modelFile = "";
  parser->parseFromFile(modelFile,
                        static_cast<int32_t>(ILogger::Severity::kWARNING));
  for (int32_t i = 0; i < parser->getNbErrors(); ++i) {
    std::cout << parser->getError(i)->desc() << std::endl;
  }
  std::unique_ptr<IBuilderConfig> config{builder->createBuilderConfig()};
  IOptimizationProfile *profile = builder->createOptimizationProfile();
  for (int i = 0; i < network->getNbInputs(); i++) {
    ITensor *input = network->getInput(i);
    Dims minDims = input->getDimensions();
    Dims optDims = input->getDimensions();
    Dims maxDims = input->getDimensions();

    minDims.d[0] = 1;
    optDims.d[0] = 4;
    maxDims.d[0] = 16;
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN,
                           minDims);
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT,
                           optDims);
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX,
                           maxDims);
  }
  config->addOptimizationProfile(profile);

  std::unique_ptr<IHostMemory> serializedModel{
      builder->buildSerializedNetwork(*network, *config)};
  std::unique_ptr<IRuntime> runtime{createInferRuntime(logger)};
  std::unique_ptr<ICudaEngine> engine{runtime->deserializeCudaEngine(
      serializedModel->data(), serializedModel->size())};
  std::unique_ptr<IExecutionContext> context{engine->createExecutionContext()};

  std::vector<void *> buffers(engine->getNbIOTensors(), nullptr);

  for (int32_t i = 0; i < network->getNbInputs(); i++) {
    ITensor *input = network->getInput(i);
    Dims dims = input->getDimensions();
    dims.d[0] = 4;
    context->setInputShape(input->getName(), dims);
  }

  for (int32_t i = 0; i < engine->getNbIOTensors(); i++) {
    auto const name = engine->getIOTensorName(i);
    Dims dims = engine->getTensorShape(name);
    uint64_t elements = 1;
    for (int j = 0; j < dims.nbDims; j++)
      elements *= dims.d[j];
    DataType type = engine->getTensorDataType(name);
    size_t bytes = elements * elemSize(type);
    cudaMalloc(&buffers[i], bytes);
    context->setTensorAddress(name, buffers[i]);
  }

  return 0;
}
