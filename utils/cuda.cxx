#include "NvInferRuntime.h"
#include <cstddef>

using namespace nvinfer1;

size_t elemSize(DataType t) {
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
