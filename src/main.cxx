#include "decoder.hxx"
#include "encoder.hxx"
#include "engine.hxx"
#include "logger.hxx"
#include "postprocess.hxx"
#include "preprocess.hxx"
#include "server.hxx"
#include "streams.hxx"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cxxopts.hpp>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

struct AppConfig {
  bool build = false;
  bool help = false;
  std::string model_path = "input/models/yolov8n.onnx";
  std::string engine_path = "output/yolov8n.engine";
  std::vector<std::string> input_urls;
  std::string output_url = "rtsp://localhost:8554/argus";
  int num_workers = 1;
};

bool parse_args(int argc, char **argv, AppConfig &config) {
  cxxopts::Options options("argus", "TensorRT YOLO live video pipeline");
  options.add_options()("build", "Build the TensorRT engine file and exit")(
      "model", "ONNX model path",
      cxxopts::value<std::string>(config.model_path)
          ->default_value(config.model_path))(
      "engine", "TensorRT engine path",
      cxxopts::value<std::string>(config.engine_path)
          ->default_value(config.engine_path))(
      "input", "Input stream URL",
      cxxopts::value<std::vector<std::string>>(config.input_urls))(
      "output", "Output stream URL",
      cxxopts::value<std::string>(config.output_url)
          ->default_value(config.output_url))(
      "workers", "The number of workers to process the streams",
      cxxopts::value<int>(config.num_workers)
          ->default_value(std::to_string(config.num_workers)))("h,help",
                                                               "Show help");

  try {
    auto result = options.parse(argc, argv);
    config.build = result.count("build") > 0;
    config.help = result.count("help") > 0;

    if (config.help) {
      std::cout << options.help() << std::endl;
      return true;
    }

  } catch (const cxxopts::exceptions::exception &err) {
    std::cerr << "argument error: " << err.what() << std::endl;
    std::cout << options.help() << std::endl;
    return false;
  }

  return true;
}

std::string indexed_output_url(const std::string &url, int stream_index) {
  std::string index = "-" + std::to_string(stream_index);
  std::size_t slash_pos = url.find_last_of("/\\");
  std::size_t dot_pos = url.rfind('.');

  if (dot_pos != std::string::npos &&
      (slash_pos == std::string::npos || dot_pos > slash_pos)) {
    return url.substr(0, dot_pos) + index + url.substr(dot_pos);
  }

  return url + index;
}

void run_decode(Decoder &decoder, ThreadQueue<StreamFrame> &decoder_queue,
                int stream_index, std::atomic<int64_t> &frame_count,
                std::atomic<bool> &stream_ok) {
  while (nextFrame(decoder)) {
    int64_t frame_id = frame_count++;
    if (frame_count == 1)
      log_msg(LOG_INFO, "main", "first frame decoded");

    AVFrame *frame = av_frame_clone(decoder.frame);
    if (!frame) {
      log_msg(LOG_ERROR, "main", "could not clone decoded frame");
      stream_ok = false;
      break;
    }

    decoder_queue.push(StreamFrame{frame, stream_index, frame_id});
  }

  decoder_queue.close();
}

void run_preprocess(Preprocessor &pre, ThreadQueue<StreamFrame> &decoder_queue,
                    ThreadQueue<StreamFrame> &preprocessor_queue) {
  std::unique_ptr<float[]> cpu_input{new float[3 * 640 * 640]};
  StreamFrame item;
  while (decoder_queue.pop(item)) {
    item.frame_rgb = preprocess(pre, item.frame, cpu_input.get());
    preprocessor_queue.push(std::move(item));
  }

  preprocessor_queue.close();
}

void run_infer(Engine &engine,
               std::vector<ThreadQueue<StreamFrame>> &preprocessor_queue,
               std::vector<ThreadQueue<StreamFrame>> &inference_queue,
               int num_workers) {
  std::unique_ptr<float[]> cpu_output{
      new float[engine.outputBytes / sizeof(float)]};

  std::vector<bool> stream_done(preprocessor_queue.size(), false);
  std::vector<StreamFrame> stream_batch;
  int active_streams = (int)preprocessor_queue.size();
  const int single_cpu_input = 3 * 640 * 640;

  std::vector<float> cpu_input;
  cpu_input.resize(num_workers * single_cpu_input);

  while (active_streams > 0) {
    bool did_work = false;
    stream_batch.clear();

    for (int i = 0; i < (int)preprocessor_queue.size(); i++) {
      if (stream_done[i])
        continue;

      StreamFrame item;
      if (preprocessor_queue[i].try_pop(item)) {
        did_work = true;
        int batch_index = (int)stream_batch.size();
        std::copy(item.frame_rgb.get(), item.frame_rgb.get() + single_cpu_input,
                  cpu_input.data() + batch_index * single_cpu_input);
        stream_batch.push_back(std::move(item));
      } else if (preprocessor_queue[i].closed_and_empty()) {
        stream_done[i] = true;
        active_streams--;
        inference_queue[i].close();
      }
    }

    if (!stream_batch.empty()) {
      std::unique_ptr<float[]> batch_output =
          run_inference(engine, cpu_input.data(), cpu_output.get(),
                        (int)stream_batch.size());
      if (!batch_output)
        continue;

      for (int i = 0; i < (int)stream_batch.size(); i++) {
        stream_batch[i].output.reset(new float[engine.outputElementsPerFrame]);
        std::copy(batch_output.get() + i * engine.outputElementsPerFrame,
                  batch_output.get() + (i + 1) * engine.outputElementsPerFrame,
                  stream_batch[i].output.get());
        inference_queue[stream_batch[i].stream_index].push(
            std::move(stream_batch[i]));
      }
    }

    if (!did_work && active_streams > 0)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

void run_postprocess(ThreadQueue<StreamFrame> &inference_queue,
                     ThreadQueue<StreamFrame> &postprocessor_queue,
                     int max_dets, int width, int height) {
  StreamFrame item;
  while (inference_queue.pop(item)) {
    item.detections = postprocess(item.output.get(), max_dets, width, height);
    postprocessor_queue.push(std::move(item));
  }

  postprocessor_queue.close();
}

void run_draw(ThreadQueue<StreamFrame> &postprocessor_queue,
              ThreadQueue<StreamFrame> &encoder_queue) {
  StreamFrame item;
  while (postprocessor_queue.pop(item)) {
    draw_box(item.detections, item.frame);
    encoder_queue.push(std::move(item));
  }

  encoder_queue.close();
}

void run_encode(Encoder &encoder, Server &server,
                ThreadQueue<StreamFrame> &encoder_queue,
                std::atomic<bool> &stream_ok) {
  StreamFrame item;
  while (encoder_queue.pop(item)) {
    if (!serve_stream(encoder, server, item.frame)) {
      log_msg(LOG_ERROR, "main", "frame encode or publish failed");
      stream_ok = false;
    }
  }
}

bool run_stream(
    Engine &engine, const AppConfig &config, int stream_index,
    std::vector<ThreadQueue<StreamFrame>> &preprocessor_shared_queue,
    std::vector<ThreadQueue<StreamFrame>> &inference_shared_queue,
    int max_batch_size) {
  const std::string &input_url = config.input_urls[stream_index];
  std::string output_url = indexed_output_url(config.output_url, stream_index);
  ThreadQueue<StreamFrame> &preprocessor_queue =
      preprocessor_shared_queue[stream_index];
  ThreadQueue<StreamFrame> &inference_queue =
      inference_shared_queue[stream_index];

  log_msg(LOG_INFO, "main",
          "stream " + std::to_string(stream_index) + " input: " + input_url);
  log_msg(LOG_INFO, "main",
          "stream " + std::to_string(stream_index) + " output: " + output_url);

  Decoder decoder;
  log_msg(LOG_INFO, "main", "input stream open started");
  if (!openStream(decoder, input_url.c_str())) {
    preprocessor_queue.close();
    inference_queue.close();
    return false;
  }
  log_msg(LOG_INFO, "main", "input stream open finished");

  Preprocessor pre;
  log_msg(LOG_INFO, "main", "preprocess setup started");
  initPreprocess(pre, decoder.codecCtx->width, decoder.codecCtx->height,
                 (int)decoder.codecCtx->pix_fmt);
  log_msg(LOG_INFO, "main", "preprocess setup finished");

  int max_dets = (int)(engine.outputBytes /
                       (sizeof(float) * max_batch_size * 6));

  log_msg(LOG_INFO, "main",
          "max detections from model output: " + std::to_string(max_dets));

  AVStream *in_stream = decoder.formatCtx->streams[decoder.videoStreamIndex];
  Encoder encoder;
  log_msg(LOG_INFO, "main", "encoder setup started");
  if (!open_encoder(encoder, in_stream, decoder.codecCtx)) {
    preprocessor_queue.close();
    inference_queue.close();
    destroyPreprocess(pre);
    closeStream(decoder);
    return false;
  }
  log_msg(LOG_INFO, "main", "encoder setup finished");

  Server server;
  log_msg(LOG_INFO, "main", "output stream setup started");
  if (!open_server(server, output_url.c_str(), encoder.codec_ctx)) {
    preprocessor_queue.close();
    inference_queue.close();
    close_encoder(encoder);
    destroyPreprocess(pre);
    closeStream(decoder);
    return false;
  }
  log_msg(LOG_INFO, "main", "output stream setup finished");

  log_msg(LOG_INFO, "main", "frame processing loop started");
  std::atomic<bool> stream_ok{true};
  std::atomic<int64_t> frame_count{0};

  ThreadQueue<StreamFrame> decoder_queue;
  ThreadQueue<StreamFrame> postprocessor_queue;
  ThreadQueue<StreamFrame> encoder_queue;

  std::thread decode_thread(run_decode, std::ref(decoder),
                            std::ref(decoder_queue), stream_index,
                            std::ref(frame_count), std::ref(stream_ok));
  std::thread preprocess_thread(run_preprocess, std::ref(pre),
                                std::ref(decoder_queue),
                                std::ref(preprocessor_queue));
  std::thread postprocess_thread(
      run_postprocess, std::ref(inference_queue), std::ref(postprocessor_queue),
      max_dets, decoder.codecCtx->width, decoder.codecCtx->height);
  std::thread draw_thread(run_draw, std::ref(postprocessor_queue),
                          std::ref(encoder_queue));
  std::thread encode_thread(run_encode, std::ref(encoder), std::ref(server),
                            std::ref(encoder_queue), std::ref(stream_ok));

  decode_thread.join();
  preprocess_thread.join();
  postprocess_thread.join();
  draw_thread.join();
  encode_thread.join();

  log_msg(LOG_INFO, "main",
          "frame loop ended after " + std::to_string(frame_count.load()) +
              " frames");
  log_msg(LOG_INFO, "main", "flushing encoder");
  flush_encoder(encoder, server);
  close_server(server);
  close_encoder(encoder);
  destroyPreprocess(pre);
  closeStream(decoder);
  log_msg(LOG_INFO, "main", "shutdown complete");
  return stream_ok;
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
    bool built = build_engine_file(config.model_path.c_str(),
                                   config.engine_path.c_str());
    log_msg(LOG_INFO, "main", "TensorRT engine build finished");
    return built ? 0 : -1;
  }

  int input_count = (int)config.input_urls.size();
  if (input_count == 0) {
    log_msg(LOG_ERROR, "main", "run mode requires at least one --input URL");
    return -1;
  }

  if (config.num_workers < 1) {
    log_msg(LOG_ERROR, "main", "--workers must be at least 1");
    return -1;
  }

  int hardware_threads = (int)std::thread::hardware_concurrency();
  int max_threads = hardware_threads > 4 ? hardware_threads - 4 : 1;
  constexpr int stage_threads_per_stream = 5;
  int max_stream_workers = std::max(1, max_threads / stage_threads_per_stream);
  int worker_count =
      std::min({input_count, config.num_workers, max_stream_workers});

  Engine engine;
  engine = load_engine(config.engine_path.c_str(), worker_count);
  log_msg(LOG_INFO, "main", "TensorRT engine load finished");

  if (!engine.context || !engine.inputBuffer || !engine.outputBuffer) {
    log_msg(LOG_ERROR, "main", "TensorRT engine is not ready");
    destroy_engine(engine);
    return -1;
  }

  std::vector<ThreadQueue<StreamFrame>> preprocessor_shared_queue(worker_count);
  std::vector<ThreadQueue<StreamFrame>> inference_shared_queue(worker_count);
  std::thread infer_thread(run_infer, std::ref(engine),
                           std::ref(preprocessor_shared_queue),
                           std::ref(inference_shared_queue), worker_count);

  std::vector<std::thread> t(worker_count);
  for (int i = 0; i < worker_count; i++) {
    t[i] = std::thread(run_stream, std::ref(engine), std::ref(config), i,
                       std::ref(preprocessor_shared_queue),
                       std::ref(inference_shared_queue), worker_count);
  }
  for (int i = 0; i < worker_count; i++)
    t[i].join();
  infer_thread.join();

  destroy_engine(engine);
  return 0;
}
