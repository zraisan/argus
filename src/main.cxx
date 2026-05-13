#include "decoder.hxx"
#include "encoder.hxx"
#include "engine.hxx"
#include "logger.hxx"
#include "postprocess.hxx"
#include "preprocess.hxx"
#include "server.hxx"
#include <cxxopts.hpp>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

struct AppConfig {
  bool build = false;
  bool help = false;
  std::string model_path = "input/models/yolov8n.onnx";
  std::string engine_path = "output/yolov8n.engine";
  std::vector<std::string> input_urls;
  std::string output_url = "rtsp://localhost:8554/argus";
  std::string rtsp_transport = "tcp";
};

bool parse_args(int argc, char **argv, AppConfig &config) {
  cxxopts::Options options("argus", "TensorRT YOLO RTSP pipeline");
  options.add_options()
      ("build", "Build the TensorRT engine file and exit")
      ("model", "ONNX model path",
       cxxopts::value<std::string>(config.model_path)
           ->default_value(config.model_path))
      ("engine", "TensorRT engine path",
       cxxopts::value<std::string>(config.engine_path)
           ->default_value(config.engine_path))
      ("input", "Input RTSP URL",
       cxxopts::value<std::vector<std::string>>(config.input_urls))
      ("output", "Output RTSP URL",
       cxxopts::value<std::string>(config.output_url)
           ->default_value(config.output_url))
      ("rtsp-transport", "Input RTSP transport: tcp or udp",
       cxxopts::value<std::string>(config.rtsp_transport)
           ->default_value(config.rtsp_transport))
      ("h,help", "Show help");

  try {
    auto result = options.parse(argc, argv);
    config.build = result.count("build") > 0;
    config.help = result.count("help") > 0;

    if (config.help) {
      std::cout << options.help() << std::endl;
      return true;
    }

    if (!config.build && config.input_urls.size() != 1) {
      std::cerr << "run mode requires exactly one --input URL" << std::endl;
      std::cout << options.help() << std::endl;
      return false;
    }

    if (config.rtsp_transport != "tcp" && config.rtsp_transport != "udp") {
      std::cerr << "--rtsp-transport must be tcp or udp" << std::endl;
      std::cout << options.help() << std::endl;
      return false;
    }
  } catch (const cxxopts::exceptions::exception &err) {
    std::cerr << "argument error: " << err.what() << std::endl;
    std::cout << options.help() << std::endl;
    return false;
  }

  return true;
}

int main(int argc, char **argv) {
  AppConfig config;
  if (!parse_args(argc, argv, config))
    return -1;
  if (config.help)
    return 0;

  log_msg(LOG_INFO, "main", "argus starting");
  log_msg(LOG_INFO, "main", std::string("model: ") + config.model_path);
  log_msg(LOG_INFO, "main", std::string("engine: ") + config.engine_path);

  if (config.build) {
    log_msg(LOG_INFO, "main", "TensorRT engine build started");
    bool built =
        build_engine_file(config.model_path.c_str(), config.engine_path.c_str());
    log_msg(LOG_INFO, "main", "TensorRT engine build finished");
    return built ? 0 : -1;
  }

  const std::string &input_url = config.input_urls[0];
  log_msg(LOG_INFO, "main", std::string("input: ") + input_url);
  log_msg(LOG_INFO, "main", std::string("output: ") + config.output_url);
  log_msg(LOG_INFO, "main",
          std::string("RTSP input transport: ") + config.rtsp_transport);

  Engine engine;
  log_msg(LOG_INFO, "main", "TensorRT engine load started");
  engine = load_engine(config.engine_path.c_str());
  log_msg(LOG_INFO, "main", "TensorRT engine load finished");

  if (!engine.context || !engine.inputBuffer || !engine.outputBuffer) {
    log_msg(LOG_ERROR, "main", "TensorRT engine is not ready");
    destroy_engine(engine);
    return -1;
  }

  Decoder decoder;
  log_msg(LOG_INFO, "main", "RTSP input open started");
  if (!openStream(decoder, input_url.c_str(), config.rtsp_transport.c_str())) {
    destroy_engine(engine);
    return -1;
  }
  log_msg(LOG_INFO, "main", "RTSP input open finished");

  Preprocessor pre;
  log_msg(LOG_INFO, "main", "preprocess setup started");
  initPreprocess(pre, decoder.codecCtx->width, decoder.codecCtx->height,
                 (int)decoder.codecCtx->pix_fmt);
  log_msg(LOG_INFO, "main", "preprocess setup finished");

  std::unique_ptr<float[]> cpu_input{new float[3 * 640 * 640]};
  int max_dets = (int)(engine.outputBytes / (sizeof(float) * 6));
  std::unique_ptr<float[]> cpu_output{
      new float[engine.outputBytes / sizeof(float)]};

  log_msg(LOG_INFO, "main",
          "max detections from model output: " + std::to_string(max_dets));

  AVStream *in_stream = decoder.formatCtx->streams[decoder.videoStreamIndex];
  Encoder encoder;
  log_msg(LOG_INFO, "main", "encoder setup started");
  if (!open_encoder(encoder, in_stream, decoder.codecCtx)) {
    destroyPreprocess(pre);
    closeStream(decoder);
    destroy_engine(engine);
    return -1;
  }
  log_msg(LOG_INFO, "main", "encoder setup finished");

  Server server;
  log_msg(LOG_INFO, "main", "RTSP output publish started");
  if (!open_server(server, config.output_url.c_str(), encoder.codec_ctx)) {
    close_encoder(encoder);
    destroyPreprocess(pre);
    closeStream(decoder);
    destroy_engine(engine);
    return -1;
  }
  log_msg(LOG_INFO, "main", "RTSP output publish finished");

  log_msg(LOG_INFO, "main", "frame processing loop started");
  int64_t frame_count = 0;
  while (nextFrame(decoder)) {
    frame_count++;
    if (frame_count == 1)
      log_msg(LOG_INFO, "main", "first frame decoded");

    preprocess(pre, decoder.frame, cpu_input.get());
    run_inference(engine, cpu_input.get(), cpu_output.get());

    std::vector<Detection> dets =
        postprocess(cpu_output.get(), max_dets, decoder.codecCtx->width,
                    decoder.codecCtx->height);
    draw_box(dets, decoder.frame);

    if (!serve_stream(encoder, server, decoder.frame)) {
      log_msg(LOG_ERROR, "main", "frame encode or publish failed");
      break;
    }

    if (frame_count == 1 || frame_count % 30 == 0) {
      log_msg(LOG_INFO, "main",
              "frame=" + std::to_string(frame_count) +
                  " pts=" + std::to_string(decoder.frame->pts) +
                  " detections=" + std::to_string(dets.size()));
    }
  }

  log_msg(LOG_INFO, "main",
          "frame loop ended after " + std::to_string(frame_count) + " frames");
  log_msg(LOG_INFO, "main", "flushing encoder");
  flush_encoder(encoder, server);
  close_server(server);
  close_encoder(encoder);
  destroyPreprocess(pre);
  closeStream(decoder);
  destroy_engine(engine);
  log_msg(LOG_INFO, "main", "shutdown complete");
  return 0;
}
