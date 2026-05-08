#include "decoder.hxx"
#include "encoder.hxx"
#include "engine.hxx"
#include "postprocess.hxx"
#include "preprocess.hxx"
#include "server.hxx"
#include <iostream>
#include <libavformat/avformat.h>
#include <memory>
#include <vector>

int main() {
  Engine engine = buildEngine("input/models/yolov8n.onnx");

  Decoder decoder;
  if (!openStream(decoder,
                  "rtsp://stream.strba.sk:1935/strba/VYHLAD_JAZERO.stream")) {
    return -1;
  }

  Preprocessor pre;
  initPreprocess(pre, decoder.codecCtx->width, decoder.codecCtx->height,
                 (int)decoder.codecCtx->pix_fmt);

  std::unique_ptr<float[]> cpuInput{new float[3 * 640 * 640]};
  std::unique_ptr<float[]> cpuOutput{
      new float[engine.outputBytes / sizeof(float)]};

  AVStream *in_stream = decoder.formatCtx->streams[decoder.videoStreamIndex];
  Encoder encoder;
  if (!open_encoder(encoder, in_stream, decoder.codecCtx)) {
    destroyPreprocess(pre);
    closeStream(decoder);
    destroyEngine(engine);
    return -1;
  }

  Server server;
  if (!open_server(server, "rtsp://localhost:8554/argus", encoder.codec_ctx)) {
    close_encoder(encoder);
    destroyPreprocess(pre);
    closeStream(decoder);
    destroyEngine(engine);
    return -1;
  }

  while (nextFrame(decoder)) {
    std::cout << "Frame pts=" << decoder.frame->pts << std::endl;
    preprocess(pre, decoder.frame, cpuInput.get());
    runInference(engine, cpuInput.get(), cpuOutput.get());
    std::vector<Detection> dets =
        postprocess(cpuOutput.get(), 100, decoder.codecCtx->width,
                    decoder.codecCtx->height);
    draw_box(dets, decoder.frame);
    if (!serve_stream(encoder, server, decoder.frame))
      break;
  }

  flush_encoder(encoder, server);
  close_server(server);
  close_encoder(encoder);
  destroyPreprocess(pre);
  closeStream(decoder);
  destroyEngine(engine);
  return 0;
}
