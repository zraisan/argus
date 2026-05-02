cmake_minimum_required(VERSION 3.23)

project(live-od)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)

add_executable(argus)

target_sources(argus
  PRIVATE
    src/main.cxx
    utils/cuda.cxx
)

target_include_directories(argus
  PRIVATE
    utils
)

target_link_libraries(argus
  PRIVATE
    CUDA::cudart
    nvinfer
    nvonnxparser
)


