cmake_minimum_required(VERSION 3.23)

project(live-od)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)

add_executable(argus)

target_sources(argus
  PRIVATE
    src/main.cxx
    src/engine.cxx
    src/decoder.cxx
    src/encoder.cxx
    src/server.cxx
    src/preprocess.cxx
    src/postprocess.cxx
    utils/cuda.cxx
)

target_include_directories(argus
  PRIVATE
    utils
)

find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBAV REQUIRED IMPORTED_TARGET
    libavformat
    libavcodec
    libswscale
    libavutil
)

target_link_libraries(argus
  PRIVATE
    CUDA::cudart
    nvinfer
    nvonnxparser
    PkgConfig::LIBAV
)
