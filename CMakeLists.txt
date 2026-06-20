cmake_minimum_required(VERSION 3.23)

project(live-od)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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
    utils/streams.cxx
    utils/logger.cxx
)

target_include_directories(argus
  PRIVATE
    utils
    libs
    libs/engine
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
    PkgConfig::LIBAV
    migraphx
)


find_package(CUDAToolkit QUIET)
find_library(TENSORRT_LIBRARY nvinfer)
find_library(TENSORRT_ONNXPARSER_LIBRARY nvonnxparser)

if (CUDAToolkit_FOUND AND TENSORRT_LIBRARY AND TENSORRT_ONNXPARSER_LIBRARY)
  target_link_libraries(argus
    PRIVATE
      CUDA::cudart
      ${TENSORRT_LIBRARY}
      ${TENSORRT_ONNXPARSER_LIBRARY}
  )
endif()


find_package(migraphx QUIET)

if(migraphx_FOUND)
  target_link_libraries(argus PRIVATE migraphx::c)
endif()

