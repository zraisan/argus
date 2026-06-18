<div align="center">

# argus

**TensorRT YOLO Live Object Detection for Video Streams**

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)
[![C++](https://img.shields.io/badge/C%2B%2B-20-00599C.svg?logo=cplusplus)](https://en.cppreference.com/w/cpp/20)
[![CMake](https://img.shields.io/badge/CMake-3.23%2B-064F8C.svg?logo=cmake)](https://cmake.org/)
[![NVIDIA](https://img.shields.io/badge/NVIDIA-TensorRT%20%7C%20NVENC-76B900.svg?logo=nvidia)](https://developer.nvidia.com/tensorrt)

</div>

argus is a C++ live object detection pipeline for video streams. It decodes an
input with FFmpeg/libav, preprocesses frames for YOLO, runs inference with
TensorRT, draws bounding boxes on the decoded frame, encodes the result with
NVIDIA NVENC or libx264, and writes the processed video to a stream output.

The project is currently focused on learning and building the full video path
step by step.

## Demo

Single stream:

<p align="center">
  <img src="docs/assets/argus_1_stream_yolo11s_demo.gif" alt="argus YOLO11s single stream demo" width="100%">
</p>

Two streams:

<p align="center">
  <img src="docs/assets/argus_2_streams_yolo11s_demo.gif" alt="argus YOLO11s two stream demo" width="100%">
</p>

Thirty-two streams:

<p align="center">
  <img src="docs/assets/argus_32_streams_yolo11s_demo.gif" alt="argus YOLO11s 32 stream demo" width="100%">
</p>

The demos use public HLS traffic cameras processed with YOLO11s. Argus decodes
the inputs, runs TensorRT inference, draws detections, and writes processed HLS
outputs.

The one-stream and two-stream demos were captured as separate runs with only
those inputs active.

All demos were captured on an NVIDIA GeForce RTX 3070 Laptop GPU with 8 GB of
VRAM.

## Features

- **Stream Input**: Opens video inputs through FFmpeg/libav, including RTSP, HLS, HTTP media, and local files
- **NVIDIA Decode Path**: Prefers CUVID decoders such as `h264_cuvid` when the input codec supports them
- **TensorRT Inference**: Builds and runs a TensorRT engine from a YOLO ONNX model
- **Frame Preprocessing**: Letterboxes decoded frames into the RGB tensor layout expected by the model
- **Postprocessing**: Converts model output into detection boxes in source-frame coordinates
- **Bounding Box Overlay**: Draws detections directly onto the local decoded video frame
- **H.264 Encoding**: Prefers `h264_nvenc` and falls back to `libx264`
- **Stream Output**: Writes encoded packets to RTSP, HLS, MP4, MPEG-TS, or Matroska outputs through FFmpeg/libav

## Pipeline

<div align="center">

```mermaid
flowchart TD
    inputs[Input URLs] --> work_q[Decode work queue]

    subgraph decode_pool[Decode Thread Pool]
      dec1[Decoder worker]
      dec2[Decoder worker]
      decn[Decoder worker]
    end

    work_q --> dec1
    work_q --> dec2
    work_q --> decn
    dec1 --> decoded_q[Decoded frame queue]
    dec2 --> decoded_q
    decn --> decoded_q

    subgraph preprocess_pool[Preprocess Thread Pool]
      pre1[Preprocess worker]
      pre2[Preprocess worker]
      pren[Preprocess worker]
    end

    decoded_q --> pre1
    decoded_q --> pre2
    decoded_q --> pren
    pre1 --> tensor_q[Preprocessed tensor queue]
    pre2 --> tensor_q
    pren --> tensor_q

    tensor_q --> batcher[Batch collector]
    batcher --> infer[Shared TensorRT inference thread]
    infer --> result_q[Inference result queue]

    subgraph post_pool[Postprocess Thread Pool]
      post1[Postprocess worker]
      post2[Postprocess worker]
      postn[Postprocess worker]
    end

    result_q --> post1
    result_q --> post2
    result_q --> postn
    post1 --> post_q[Postprocessed frame queue]
    post2 --> post_q
    postn --> post_q

    subgraph draw_pool[Draw Thread Pool]
      draw1[Draw worker]
      draw2[Draw worker]
      drawn[Draw worker]
    end

    post_q --> draw1
    post_q --> draw2
    post_q --> drawn
    draw1 --> encode_q[Encode queue]
    draw2 --> encode_q
    drawn --> encode_q

    subgraph encode_pool[Encode Thread Pool]
      enc1[Encode worker]
      enc2[Encode worker]
      encn[Encode worker]
    end

    encode_q --> enc1
    encode_q --> enc2
    encode_q --> encn
    enc1 --> state[Per-stream encoder and muxer state]
    enc2 --> state
    encn --> state
    state --> order[Frame ordering by frame_id]
    order --> outputs[Stream outputs]
```

</div>

Decode opens one FFmpeg decoder per input stream. Later stages use shared
queues and stage thread pools. TensorRT receives batches up to the configured
engine batch size, and encode keeps per-stream output state so each stream is
written in frame order.

### Thread Allocation

Argus currently chooses stage thread counts automatically from
`std::thread::hardware_concurrency()`.

```text
reserved_threads =
  4 when hardware_threads >= 16
  2 when hardware_threads >= 8
  1 otherwise

inference_threads = 1
max_threads = hardware_threads - reserved_threads - inference_threads
```

The remaining worker threads are split across pipeline stages:

```text
decode      40%
preprocess  10%
postprocess 10%
draw         10%
encode      30%
```

Each stage keeps at least one thread because the configured defaults start at
`1`. On a 24-thread machine, the current split is:

```text
hardware_threads = 24
reserved_threads = 4
inference_threads = 1
max_threads = 19

decode      7 threads
preprocess  1 thread
postprocess 1 thread
draw        1 thread
encode      5 threads
inference   1 thread
```

The percentage math uses integer division, so unused remainder threads are not
assigned yet.

### Stage Details

Decode scheduling:

```mermaid
flowchart LR
    urls[Input URLs] --> states[DecodeState per stream]
    urls --> work_q[Decode work queue: stream index]
    work_q --> worker[Decoder worker]
    states --> worker
    worker --> opened{Stream opened?}
    opened -- no --> open_stream[openStream and read frame rate]
    opened -- yes --> next_frame[nextFrame]
    open_stream --> next_frame
    next_frame -- decoded frame --> clone[av_frame_clone]
    clone --> stream_frame[StreamFrame: frame, stream index, frame id, fps]
    stream_frame --> decoded_q[Decoded frame queue]
    stream_frame --> requeue[Requeue stream index]
    requeue --> work_q
    next_frame -- EOF or error --> close_stream[closeStream]
    close_stream --> active_count[Decrease active stream count]
    active_count --> maybe_close[Close work queue when all streams end]
```

Preprocess:

```mermaid
flowchart LR
    decoded_q[Decoded frame queue] --> pre_worker[Preprocess worker]
    pre_worker --> format_check{Frame size or format changed?}
    format_check -- yes --> init[Create sws context]
    format_check -- no --> letterbox
    init --> letterbox[Compute letterbox scale and padding]
    letterbox --> rgb[sws_scale into RGB buffer]
    rgb --> chw[Normalize RGB to CHW float tensor]
    chw --> attach[Attach letterbox metadata to StreamFrame]
    attach --> tensor_q[Preprocessed tensor queue]
```

Inference:

```mermaid
flowchart LR
    tensor_q[Preprocessed tensor queue] --> batcher[Batch collector]
    batcher --> fill[Copy frame tensors into batched CPU input]
    fill --> infer[run_inference with current batch size]
    infer --> trt[TensorRT execution context]
    trt --> output[Batch output tensor]
    output --> split[Split output back per StreamFrame]
    split --> result_q[Inference result queue]
```

Postprocess:

```mermaid
flowchart LR
    result_q[Inference result queue] --> post_worker[Postprocess worker]
    post_worker --> parse[Read YOLO NMS output: max 300 rows]
    parse --> undo[Undo letterbox scale and padding]
    undo --> clamp[Clamp boxes to source frame]
    clamp --> dets[Detection vector]
    dets --> post_q[Postprocessed frame queue]
```

Draw:

```mermaid
flowchart LR
    post_q[Postprocessed frame queue] --> draw_worker[Draw worker]
    draw_worker --> writable[av_frame_make_writable]
    writable --> pixfmt{Pixel format}
    pixfmt -- nv12 --> draw_nv12[Draw YUV boxes into NV12 planes]
    pixfmt -- yuv420p --> draw_yuv420p[Draw YUV boxes into YUV420P planes]
    draw_nv12 --> encode_q[Encode queue]
    draw_yuv420p --> encode_q
```

Encode and output:

```mermaid
flowchart LR
    encode_q[Encode queue] --> enc_worker[Encode worker]
    enc_worker --> state[Per-stream OutputState]
    state --> pending[Store frame by frame id]
    pending --> order{Next frame id ready?}
    order -- no --> wait[Keep pending]
    order -- yes --> opened{Output opened?}
    opened -- no --> open_encoder[Open h264_nvenc or fall back to libx264]
    open_encoder --> open_server[Open output muxer]
    opened -- yes --> serve
    open_server --> serve[Encode frame and write packet]
    serve --> advance[Advance next frame id]
    advance --> order
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
- An RTSP server for RTSP output, such as MediaMTX

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

After exporting the YOLO ONNX model, build a TensorRT engine file:

```bash
./build/argus --build \
  --model input/models/yolo11s_dynamic_simplify_nms.onnx \
  --engine output/yolo11s_dynamic_simplify_nms.engine \
  --batch 16
```

Run the pipeline from an existing engine file:

```bash
./build/argus \
  --engine output/yolo11s_dynamic_simplify_nms.engine \
  --input rtsp://source \
  --output rtsp://localhost:8554/argus
```

## Output

The output type is selected from the `--output` URL:

```text
rtsp://localhost:8554/argus  RTSP
output/argus.m3u8            HLS
output/argus.mp4             MP4
output/argus.ts              MPEG-TS
output/argus.mkv             Matroska
output/argus.webm            WebM
```

For RTSP output, argus publishes to:

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

For HLS output, argus writes a playlist and segment files:

```bash
./build/argus \
  --engine output/yolo11s_dynamic_simplify_nms.engine \
  --input rtsp://source \
  --output output/argus.m3u8
```

The generated playlist can be checked with:

```bash
ffprobe output/argus.m3u8
```

## Source Layout

```text
src/
  decoder.*      input stream and packet-to-frame decoding
  preprocess.*   Frame conversion for YOLO input
  engine.*       TensorRT engine build and inference
  postprocess.*  Detection parsing and bounding box drawing
  encoder.*      AVFrame-to-H.264 packet encoding
  server.*       output muxing and packet writing
  main.cxx       Pipeline wiring

utils/
  streams.*      stream and FFmpeg helpers
  logger.*       timestamped logging

libs/
  cxxopts.hpp    command line option parsing
```

## Model Export

The current postprocess path expects a YOLO ONNX model with NMS included and an
output tensor shaped like:

```text
[batch, 300, 6]
```

Preprocessing keeps the source aspect ratio with letterbox padding. Postprocess
uses the recorded scale and padding to map model boxes back to source-frame
coordinates.

For Ultralytics YOLO11s, export with:

```bash
yolo export model=yolo11s.pt format=onnx imgsz=640 opset=17 dynamic=True nms=True simplify=True max_det=300 batch=16
```

The `simplify=True` export has been tested with TensorRT. The non-simplified
NMS graph can build but fail at inference in TensorRT shape calculation.

## Development Notes

The frame overlay code writes directly into decoded YUV buffers. It currently
supports `nv12`, `yuv420p`, and `yuvj420p`. The NVIDIA CUVID H.264 path commonly
outputs `nv12`. If a stream decodes to another pixel format, the frame must be
converted before drawing and encoding, or the drawing path must be updated for
that format.

The encoder and output muxer are separated:

```text
encoder = raw AVFrame to compressed AVPacket
server  = compressed AVPacket to stream output
```

This separation keeps the video compression step independent from the output
transport step.

## Contributing

Contributions are welcome.

### Areas to Improve

1. Improve runtime performance.
2. Add AMD GPU support through ROCm.
3. Add multi-input stream support.
4. Expand stream input and output coverage, including webcams and additional network protocols.
5. Add structured logging for FFmpeg and TensorRT errors.
6. Add small integration tests for decoder, encoder, and muxer setup.

### Style

- Keep code C++20.
- Prefer clear ownership of FFmpeg objects in small structs.
- Keep decode, inference, encode, and output responsibilities separated.
- Check FFmpeg return values and print readable errors with `av_strerror`.
- Avoid adding unrelated refactors when changing one part of the pipeline.

## License

GNU Affero General Public License v3.0. See [LICENSE](LICENSE) for the full
text.
