#include "decoder.hxx"
#include "engine.hxx"
#include "postprocess.hxx"
#include "preprocess.hxx"
#include <ios>
#include <iostream>
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

  while (nextFrame(decoder)) {
    std::cout << "Frame pts=" << decoder.frame->pts << std::endl;
    preprocess(pre, decoder.frame, cpuInput.get());
    runInference(engine, cpuInput.get(), cpuOutput.get());
    // postprocess(cpuOutput.get(), ...) — TODO
    std::vector<Detection> dets =
        postprocess(cpuOutput.get(), 100, decoder.codecCtx->width,
                    decoder.codecCtx->height);
    // encode/draw/output     — TODO
  }

  destroyPreprocess(pre);
  closeStream(decoder);
  destroyEngine(engine);
  return 0;
}
