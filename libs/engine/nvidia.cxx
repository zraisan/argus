#include "nvidia.hxx"
#include "NvOnnxParser.h"
#include "logger.hxx"
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <vector>

using namespace nvinfer1;
using namespace nvonnxparser;

namespace {
class TRTLogger : public ILogger {
  void log(Severity severity, const char *msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      LogLevel level = severity <= Severity::kERROR ? LOG_ERROR : LOG_WARNING;
      log_msg(level, "tensorrt", msg);
    }
  }
} logger;

std::string dims_string(Dims dims) {
  std::ostringstream out;
  for (int i = 0; i < dims.nbDims; i++) {
    if (i > 0)
      out << "x";
    out << dims.d[i];
  }
  return out.str();
}

const char *data_type_name(DataType type) {
  switch (type) {
  case DataType::kFLOAT:
    return "fp32";
  case DataType::kHALF:
    return "fp16";
  case DataType::kINT8:
    return "int8";
  case DataType::kINT32:
    return "int32";
  case DataType::kBOOL:
    return "bool";
  case DataType::kUINT8:
    return "uint8";
  case DataType::kFP8:
    return "fp8";
  case DataType::kBF16:
    return "bf16";
  case DataType::kINT64:
    return "int64";
  case DataType::kINT4:
    return "int4";
  case DataType::kFP4:
    return "fp4";
  }
  return "unknown";
}

std::size_t elem_size(DataType t) {
  switch (t) {
  case DataType::kFLOAT:
    return 4;
  case DataType::kHALF:
    return 2;
  case DataType::kBF16:
    return 2;
  case DataType::kINT32:
    return 4;
  case DataType::kINT64:
    return 8;
  case DataType::kINT8:
    return 1;
  case DataType::kBOOL:
    return 1;
  case DataType::kFP8:
    return 1;
  default:
    return 0;
  }
}

bool write_engine_file(const char *engine_path, const void *data,
                       std::size_t size) {
  if (!engine_path || engine_path[0] == '\0')
    return false;

  try {
    std::filesystem::path engine_file(engine_path);
    std::filesystem::path parent = engine_file.parent_path();
    if (!parent.empty())
      std::filesystem::create_directories(parent);
  } catch (const std::filesystem::filesystem_error &err) {
    log_msg(LOG_WARNING, "engine",
            std::string("could not create TensorRT output directory: ") +
                err.what());
    return false;
  }

  std::ofstream file(engine_path, std::ios::binary | std::ios::trunc);
  if (!file) {
    log_msg(LOG_WARNING, "engine",
            std::string("could not create TensorRT engine file: ") +
                engine_path);
    return false;
  }

  file.write(static_cast<const char *>(data),
             static_cast<std::streamsize>(size));
  if (!file) {
    log_msg(LOG_WARNING, "engine", "could not write TensorRT engine file");
    return false;
  }

  log_msg(LOG_INFO, "engine",
          std::string("saved TensorRT engine file: ") + engine_path);
  return true;
}

} // namespace

bool build_engine_file(const char *onnx_path, const char *engine_path,
                       int batch_size) {
  log_msg(LOG_INFO, "engine",
          std::string("creating TensorRT builder for ") + onnx_path);
  std::unique_ptr<IBuilder> builder{createInferBuilder(logger)};
  if (!builder) {
    log_msg(LOG_CRITICAL, "engine", "failed to create TensorRT builder");
    return false;
  }

  std::unique_ptr<INetworkDefinition> network{builder->createNetworkV2(
      1U << static_cast<uint32_t>(
          NetworkDefinitionCreationFlag::kSTRONGLY_TYPED))};
  if (!network) {
    log_msg(LOG_CRITICAL, "engine", "failed to create TensorRT network");
    return false;
  }

  std::unique_ptr<IParser> parser{createParser(*network, logger)};
  if (!parser) {
    log_msg(LOG_CRITICAL, "engine", "failed to create ONNX parser");
    return false;
  }

  {
    log_msg(LOG_INFO, "engine", "ONNX parse started");
    auto start = std::chrono::steady_clock::now();
    bool parsed = parser->parseFromFile(
        onnx_path, static_cast<int32_t>(ILogger::Severity::kWARNING));
    log_duration(LOG_INFO, "engine", "ONNX parse", start);

    for (int32_t i = 0; i < parser->getNbErrors(); ++i)
      log_msg(LOG_ERROR, "engine", parser->getError(i)->desc());

    if (!parsed || parser->getNbErrors() > 0) {
      log_msg(LOG_CRITICAL, "engine", "ONNX parse failed");
      return false;
    }
  }

  std::unique_ptr<IBuilderConfig> config{builder->createBuilderConfig()};
  if (!config) {
    log_msg(LOG_CRITICAL, "engine", "failed to create builder config");
    return false;
  }

  IOptimizationProfile *profile = builder->createOptimizationProfile();
  if (!profile) {
    log_msg(LOG_CRITICAL, "engine", "failed to create optimization profile");
    return false;
  }

  log_msg(LOG_INFO, "engine", "configuring optimization profile");
  for (int32_t i = 0; i < network->getNbInputs(); i++) {
    ITensor *input = network->getInput(i);
    Dims input_dims = input->getDimensions();
    Dims min_dims = input_dims, opt_dims = input_dims, max_dims = input_dims;

    min_dims.d[0] = 1;
    opt_dims.d[0] = std::max(1, batch_size / 2);
    max_dims.d[0] = batch_size;
    if (input_dims.nbDims >= 4) {
      min_dims.d[2] = opt_dims.d[2] = max_dims.d[2] = 640;
      min_dims.d[3] = opt_dims.d[3] = max_dims.d[3] = 640;
    }

    profile->setDimensions(input->getName(), OptProfileSelector::kMIN,
                           min_dims);
    profile->setDimensions(input->getName(), OptProfileSelector::kOPT,
                           opt_dims);
    profile->setDimensions(input->getName(), OptProfileSelector::kMAX,
                           max_dims);

    log_msg(LOG_INFO, "engine",
            std::string("input profile ") + input->getName() + " min=" +
                dims_string(min_dims) + " opt=" + dims_string(opt_dims) +
                " max=" + dims_string(max_dims));
  }
  config->addOptimizationProfile(profile);

  std::unique_ptr<IHostMemory> serialized_model;
  {
    log_msg(LOG_INFO, "engine", "TensorRT serialized network build started");
    auto start = std::chrono::steady_clock::now();
    serialized_model.reset(builder->buildSerializedNetwork(*network, *config));
    log_duration(LOG_INFO, "engine", "TensorRT serialized network build",
                 start);
  }
  if (!serialized_model) {
    log_msg(LOG_CRITICAL, "engine",
            "failed to build serialized TensorRT network");
    return false;
  }

  return write_engine_file(engine_path, serialized_model->data(),
                           serialized_model->size());
}

Engine load_engine(const char *engine_path, int batch_size) {
  Engine e;
  e.maxBatchSize = batch_size;
  std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
  if (!file) {
    log_msg(LOG_CRITICAL, "engine",
            std::string("could not open TensorRT engine file: ") + engine_path);
    return e;
  }

  std::streamsize size = file.tellg();
  if (size <= 0) {
    log_msg(LOG_CRITICAL, "engine", "TensorRT engine file is empty");
    return e;
  }

  std::vector<char> engine_data(static_cast<std::size_t>(size));
  file.seekg(0, std::ios::beg);
  if (!file.read(engine_data.data(), size)) {
    log_msg(LOG_CRITICAL, "engine", "could not read TensorRT engine file");
    return e;
  }

  log_msg(LOG_INFO, "engine",
          "loaded TensorRT engine file bytes=" +
              std::to_string(engine_data.size()));

  log_msg(LOG_INFO, "engine", "creating TensorRT runtime");
  e.runtime.reset(createInferRuntime(logger));
  if (!e.runtime) {
    log_msg(LOG_CRITICAL, "engine", "failed to create TensorRT runtime");
    return e;
  }

  {
    log_msg(LOG_INFO, "engine", "TensorRT engine deserialize started");
    auto start = std::chrono::steady_clock::now();
    e.engine.reset(e.runtime->deserializeCudaEngine(engine_data.data(),
                                                    engine_data.size()));
    log_duration(LOG_INFO, "engine", "TensorRT engine deserialize", start);
  }
  if (!e.engine) {
    log_msg(LOG_CRITICAL, "engine", "failed to deserialize TensorRT engine");
    return e;
  }

  log_msg(LOG_INFO, "engine", "creating execution context");
  e.context.reset(e.engine->createExecutionContext());
  if (!e.context) {
    log_msg(LOG_CRITICAL, "engine", "failed to create execution context");
    return e;
  }

  for (int32_t i = 0; i < e.engine->getNbIOTensors(); i++) {
    auto const name = e.engine->getIOTensorName(i);
    if (e.engine->getTensorIOMode(name) != TensorIOMode::kINPUT)
      continue;

    Dims dims = e.engine->getTensorShape(name);
    dims.d[0] = batch_size;
    if (dims.nbDims >= 4) {
      dims.d[2] = 640;
      dims.d[3] = 640;
    }

    if (!e.context->setInputShape(name, dims)) {
      log_msg(LOG_CRITICAL, "engine",
              std::string("failed to set input shape for ") + name);
      return e;
    }
    log_msg(LOG_INFO, "engine",
            std::string("input shape ") + name + "=" + dims_string(dims));
  }

  log_msg(LOG_INFO, "engine", "allocating tensor buffers");
  e.gpuBuffers.assign(e.engine->getNbIOTensors(), nullptr);
  for (int32_t i = 0; i < e.engine->getNbIOTensors(); i++) {
    auto const name = e.engine->getIOTensorName(i);
    Dims dims = e.context->getTensorShape(name);
    uint64_t elements = 1;
    for (int j = 0; j < dims.nbDims; j++)
      elements *= dims.d[j];
    DataType type = e.engine->getTensorDataType(name);
    std::size_t bytes = elements * elem_size(type);
    cudaMalloc(&e.gpuBuffers[i], bytes);
    e.context->setTensorAddress(name, e.gpuBuffers[i]);

    log_msg(LOG_INFO, "engine",
            std::string("tensor ") + name + " shape=" + dims_string(dims) +
                " type=" + data_type_name(type) +
                " bytes=" + std::to_string(bytes));

    if (e.engine->getTensorIOMode(name) == TensorIOMode::kINPUT) {
      e.inputName = name;
      e.inputBuffer = e.gpuBuffers[i];
      e.inputBytes = bytes;
      e.inputElementsPerFrame = elements / batch_size;
    } else {
      e.outputName = name;
      e.outputBuffer = e.gpuBuffers[i];
      e.outputBytes = bytes;
      e.outputElementsPerFrame = elements / batch_size;
    }
  }

  cudaStreamCreate(&e.stream);
  log_msg(LOG_INFO, "engine", "CUDA stream created");

  return e;
}

std::unique_ptr<float[]> run_inference(Engine &e, const float *cpu_input,
                                       float *cpu_output, int batch_size) {
  if (batch_size < 1 || batch_size > e.maxBatchSize) {
    log_msg(LOG_ERROR, "engine",
            "invalid inference batch size=" + std::to_string(batch_size) +
                " max=" + std::to_string(e.maxBatchSize));
    return nullptr;
  }

  if (e.inputName.empty() || e.outputName.empty()) {
    log_msg(LOG_ERROR, "engine", "TensorRT tensor names are not initialized");
    return nullptr;
  }

  Dims input_dims = e.context->getTensorShape(e.inputName.c_str());
  input_dims.d[0] = batch_size;
  if (!e.context->setInputShape(e.inputName.c_str(), input_dims)) {
    log_msg(LOG_ERROR, "engine", "failed to set runtime input batch shape");
    return nullptr;
  }

  std::size_t input_bytes =
      batch_size * e.inputElementsPerFrame * sizeof(float);
  cudaMemcpyAsync(e.inputBuffer, cpu_input, input_bytes, cudaMemcpyHostToDevice,
                  e.stream);

  e.context->enqueueV3(e.stream);

  Dims output_dims = e.context->getTensorShape(e.outputName.c_str());
  std::size_t output_elements = 1;
  for (int i = 0; i < output_dims.nbDims; i++)
    output_elements *= output_dims.d[i];
  std::size_t output_bytes = output_elements * sizeof(float);

  cudaMemcpyAsync(cpu_output, e.outputBuffer, output_bytes,
                  cudaMemcpyDeviceToHost, e.stream);
  cudaStreamSynchronize(e.stream);

  std::unique_ptr<float[]> output{new float[output_elements]};
  std::copy(cpu_output, cpu_output + output_elements, output.get());

  return output;
}

void destroy_engine(Engine &e) {
  log_msg(LOG_INFO, "engine", "destroying TensorRT resources");
  for (void *buf : e.gpuBuffers)
    if (buf)
      cudaFree(buf);
  e.gpuBuffers.clear();
  if (e.stream)
    cudaStreamDestroy(e.stream);
}
