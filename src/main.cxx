#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "cuda.hxx"
#include <NvInferRuntime.h>
#include <NvInferRuntimeBase.h>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <iostream>
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
  IBuilder *builder = createInferBuilder(logger);
  INetworkDefinition *network = builder->createNetworkV2(
      1U << static_cast<uint32_t>(
          NetworkDefinitionCreationFlag::kSTRONGLY_TYPED));
  IParser *parser = createParser(*network, logger);
  const char *modelFile = "";
  parser->parseFromFile(modelFile,
                        static_cast<int32_t>(ILogger::Severity::kWARNING));
  for (int32_t i = 0; i < parser->getNbErrors(); ++i) {
    std::cout << parser->getError(i)->desc() << std::endl;
  }
  IBuilderConfig *config = builder->createBuilderConfig();
  IHostMemory *serializedModel =
      builder->buildSerializedNetwork(*network, *config);
  IRuntime *runtime = createInferRuntime(logger);
  ICudaEngine *engine = runtime->deserializeCudaEngine(serializedModel->data(),
                                                       serializedModel->size());
  IExecutionContext *context = engine->createExecutionContext();

  std::vector<void *> buffers(engine->getNbIOTensors(), nullptr);
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

  delete parser;
  delete network;
  delete config;
  delete builder;
  delete serializedModel;
  return 0;
}
