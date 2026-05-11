<div align="center">

# argus

**TensorRT YOLO Live Object Detection for RTSP Streams**

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)
[![C++](https://img.shields.io/badge/C%2B%2B-20-00599C.svg?logo=cplusplus)](https://en.cppreference.com/w/cpp/20)
[![CMake](https://img.shields.io/badge/CMake-3.23%2B-064F8C.svg?logo=cmake)](https://cmake.org/)
[![NVIDIA](https://img.shields.io/badge/NVIDIA-TensorRT%20%7C%20NVENC-76B900.svg?logo=nvidia)](https://developer.nvidia.com/tensorrt)

</div>

argus is a C++ live object detection pipeline for RTSP video. It decodes an
input stream with FFmpeg/libav, preprocesses frames for YOLO, runs inference
with TensorRT, draws bounding boxes on the decoded frame, encodes the result
with NVIDIA NVENC, and publishes the processed stream to an RTSP endpoint.

The project is currently focused on learning and building the full video path
step by step.

## Features

- **RTSP Input**: Opens live RTSP video through FFmpeg/libav
- **NVIDIA Decode Path**: Prefers CUVID decoders such as `h264_cuvid` when the input codec supports them
- **TensorRT Inference**: Builds and runs a TensorRT engine from a YOLO ONNX model
- **Frame Preprocessing**: Converts decoded frames into the RGB tensor layout expected by the model
- **Postprocessing**: Converts model output into detection boxes in source-frame coordinates
- **Bounding Box Overlay**: Draws detections directly onto the local decoded video frame
- **NVIDIA Encoding**: Encodes processed frames through `h264_nvenc`
- **RTSP Publishing**: Writes encoded packets to an RTSP output URL through FFmpeg/libav

## Pipeline

```mermaid
flowchart TD
    input[RTSP input] --> decoder[FFmpeg decoder]
    decoder --> frame[AVFrame]
    frame --> preprocess[Preprocess]
    preprocess --> trt[TensorRT YOLO]
    trt --> postprocess[Postprocess detections]
    postprocess --> draw[Draw bounding boxes]
    draw --> encoder[NVIDIA H.264 encoder]
    encoder --> server[RTSP publisher]
    server --> output[RTSP output]
```

## Requirements

- CMake 3.23 or later
- A C++20 capable compiler
- NVIDIA GPU with CUDA support
- CUDA Toolkit
- TensorRT
- FFmpeg development libraries:
  - `libavformat`
  - `libavcodec`
  - `libswscale`
  - `libavutil`
- An RTSP server for publishing output, such as MediaMTX

The current encoder path expects FFmpeg to provide:

```text
h264_nvenc
```

The decoder path attempts to use NVIDIA CUVID decoders when available, for
example:

```text
h264_cuvid
hevc_cuvid
av1_cuvid
```

## Build

```bash
cmake -B build
cmake --build build
```

Build a TensorRT engine file:

```bash
./build/argus --build \
  --model input/models/yolov8n.onnx \
  --engine output/yolov8n.engine
```

Run the RTSP pipeline from an existing engine file:

```bash
./build/argus \
  --engine output/yolov8n.engine \
  --input rtsp://source \
  --output rtsp://localhost:8554/argus
```

## RTSP Output

argus publishes to:

```text
rtsp://localhost:8554/argus
```

An RTSP server must already be running on `localhost:8554`. FFmpeg's RTSP
muxer publishes packets to that server. It does not replace a full RTSP server
process by itself.

With MediaMTX running locally, viewers can open:

```text
rtsp://localhost:8554/argus
```

## Source Layout

```text
src/
  decoder.*      RTSP input and packet-to-frame decoding
  preprocess.*   Frame conversion for YOLO input
  engine.*       TensorRT engine build and inference
  postprocess.*  Detection parsing and bounding box drawing
  encoder.*      AVFrame-to-H.264 packet encoding
  server.*       RTSP output muxing and packet writing
  main.cxx       Pipeline wiring

utils/
  ffmpeg_utils.* FFmpeg error helpers
  logger.*       timestamped logging

libs/
  cxxopts.hpp    command line option parsing
```

## Development Notes

The current frame overlay code writes directly into planar YUV frame buffers.
That matches `yuv420p` style frames, but not every pixel format. If the NVIDIA
decode path produces frames in a different format, the frame must be converted
before drawing and encoding, or the drawing path must be updated for that
format.

The encoder and RTSP publisher are separated:

```text
encoder = raw AVFrame to compressed AVPacket
server  = compressed AVPacket to RTSP output
```

This separation keeps the video compression step independent from the output
transport step.

## Contributing

Contributions are welcome.

### Areas to Improve

1. Add robust pixel-format conversion for CUDA, NV12, and YUV420P frames.
2. Move preprocessing and drawing closer to the GPU path where appropriate.
3. Add multi-input stream support.
4. Add structured logging for FFmpeg and TensorRT errors.
5. Add small integration tests for decoder, encoder, and muxer setup.

### Style

- Keep code C++20.
- Prefer clear ownership of FFmpeg objects in small structs.
- Keep decode, inference, encode, and output responsibilities separated.
- Check FFmpeg return values and print readable errors with `av_strerror`.
- Avoid adding unrelated refactors when changing one part of the pipeline.

## License

GNU Affero General Public License v3.0. See [LICENSE](LICENSE) for the full
text.
