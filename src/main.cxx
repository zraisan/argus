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
#include <map>
#include <memory>
#include <mutex>
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
  int decoder_threads_count = 1;
  int preprocessor_threads_count = 1;
  int postprocessor_threads_count = 1;
  int drawer_threads_count = 1;
  int encoder_threads_count = 1;
  int max_batch_size = 16;
};

struct OutputState {
  Encoder encoder;
  Server server;
  bool output_opened = false;
  int64_t next_frame_id = 0;
  std::map<int64_t, StreamFrame> pending_frames;
  std::mutex mutex;
};

bool parse_args(int argc, char **argv, AppConfig &config) {
  cxxopts::Options options("argus", "TensorRT YOLO live video pipeline");
  options.add_options()("build", "Build the TensorRT engine file and exit")(
      "model", "ONNX model path",
      cxxopts::value<std::string>(config.model_path)
          ->default_value(config.model_path))(
      "batch", "Maximum batch size supported by the engine",
      cxxopts::value<int>(config.max_batch_size)
          ->default_value(std::to_string(config.max_batch_size)))(
      "engine", "TensorRT engine path",
      cxxopts::value<std::string>(config.engine_path)
          ->default_value(config.engine_path))(
      "input", "Input stream URL",
      cxxopts::value<std::vector<std::string>>(config.input_urls))(
      "output", "Output stream URL",
      cxxopts::value<std::string>(config.output_url)
          ->default_value(config.output_url))("h,help", "Show help");

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

bool sane_frame_rate(AVRational frame_rate) {
  if (frame_rate.num <= 0 || frame_rate.den <= 0)
    return false;

  double value = av_q2d(frame_rate);
  return value >= 1.0 && value <= 120.0;
}

AVRational decoder_frame_rate(Decoder &decoder) {
  AVStream *stream = decoder.formatCtx->streams[decoder.videoStreamIndex];
  if (stream && sane_frame_rate(stream->avg_frame_rate))
    return stream->avg_frame_rate;

  if (stream && sane_frame_rate(stream->r_frame_rate))
    return stream->r_frame_rate;

  return AVRational{30, 1};
}

void run_decode(const AppConfig &config, std::atomic<int> &next_stream_index,
                ThreadQueue<StreamFrame> &decoder_queue,
                std::atomic<bool> &pipeline_ok) {
  int stream_count = (int)config.input_urls.size();

  while (true) {
    int stream_index = next_stream_index++;
    if (stream_index >= stream_count)
      break;

    const std::string &input_url = config.input_urls[stream_index];
    log_msg(LOG_INFO, "decoder",
            "stream " + std::to_string(stream_index) + " input: " + input_url);

    Decoder decoder;
    if (!openStream(decoder, input_url.c_str())) {
      pipeline_ok = false;
      continue;
    }

    AVRational frame_rate = decoder_frame_rate(decoder);
    int64_t frame_count = 0;

    while (nextFrame(decoder)) {
      int64_t frame_id = frame_count++;
      if (frame_count == 1)
        log_msg(LOG_INFO, "decoder",
                "stream " + std::to_string(stream_index) +
                    " first frame decoded");

      AVFrame *frame = av_frame_clone(decoder.frame);
      if (!frame) {
        log_msg(LOG_ERROR, "decoder", "could not clone decoded frame");
        pipeline_ok = false;
        break;
      }

      if (!decoder_queue.push(
              StreamFrame{frame, stream_index, frame_id, frame_rate})) {
        av_frame_free(&frame);
        break;
      }
    }

    closeStream(decoder);
    log_msg(LOG_INFO, "decoder",
            "stream " + std::to_string(stream_index) +
                " decoded frames=" + std::to_string(frame_count));
  }
}

void run_preprocess(ThreadQueue<StreamFrame> &decoder_queue,
                    ThreadQueue<StreamFrame> &preprocessor_queue) {
  Preprocessor pre;
  bool pre_ready = false;
  int src_w = 0;
  int src_h = 0;
  int src_pix_fmt = -1;
  std::unique_ptr<float[]> cpu_input{new float[3 * 640 * 640]};
  StreamFrame item;
  while (decoder_queue.pop(item)) {
    if (!pre_ready || src_w != item.frame->width ||
        src_h != item.frame->height || src_pix_fmt != item.frame->format) {
      if (pre_ready)
        destroyPreprocess(pre);

      src_w = item.frame->width;
      src_h = item.frame->height;
      src_pix_fmt = item.frame->format;
      initPreprocess(pre, src_w, src_h, src_pix_fmt);
      pre_ready = true;
    }

    item.frame_rgb = preprocess(pre, item.frame, cpu_input.get());
    preprocessor_queue.push(std::move(item));
  }

  if (pre_ready)
    destroyPreprocess(pre);
}

void run_infer(Engine &engine, ThreadQueue<StreamFrame> &preprocessor_queue,
               ThreadQueue<StreamFrame> &inference_queue, int batch_capacity) {
  std::unique_ptr<float[]> cpu_output{
      new float[engine.outputBytes / sizeof(float)]};

  std::vector<StreamFrame> stream_batch;
  const int single_cpu_input = 3 * 640 * 640;

  std::vector<float> cpu_input;
  cpu_input.resize(batch_capacity * single_cpu_input);
  stream_batch.reserve(batch_capacity);

  int64_t batch_count = 0;
  int last_logged_batch_size = -1;
  log_msg(LOG_INFO, "infer",
          "shared inference thread started max_batch=" +
              std::to_string(batch_capacity));

  StreamFrame item;
  while (preprocessor_queue.pop(item)) {
    stream_batch.clear();

    int batch_index = (int)stream_batch.size();
    std::copy(item.frame_rgb.get(), item.frame_rgb.get() + single_cpu_input,
              cpu_input.data() + batch_index * single_cpu_input);
    stream_batch.push_back(std::move(item));

    while ((int)stream_batch.size() < batch_capacity) {
      StreamFrame next_item;
      if (!preprocessor_queue.try_pop(next_item))
        break;

      batch_index = (int)stream_batch.size();
      std::copy(next_item.frame_rgb.get(),
                next_item.frame_rgb.get() + single_cpu_input,
                cpu_input.data() + batch_index * single_cpu_input);
      stream_batch.push_back(std::move(next_item));
    }

    int current_batch_size = (int)stream_batch.size();
    if (current_batch_size != last_logged_batch_size || batch_count % 60 == 0) {
      log_msg(LOG_INFO, "infer",
              "batch=" + std::to_string(batch_count) +
                  " size=" + std::to_string(current_batch_size));
      last_logged_batch_size = current_batch_size;
    }

    std::unique_ptr<float[]> batch_output = run_inference(
        engine, cpu_input.data(), cpu_output.get(), current_batch_size);
    if (!batch_output)
      continue;

    for (int i = 0; i < current_batch_size; i++) {
      stream_batch[i].output.reset(new float[engine.outputElementsPerFrame]);
      std::copy(batch_output.get() + i * engine.outputElementsPerFrame,
                batch_output.get() + (i + 1) * engine.outputElementsPerFrame,
                stream_batch[i].output.get());
      inference_queue.push(std::move(stream_batch[i]));
    }
    batch_count++;
  }

  log_msg(LOG_INFO, "infer", "shared inference thread stopped");
}

void run_postprocess(ThreadQueue<StreamFrame> &inference_queue,
                     ThreadQueue<StreamFrame> &postprocessor_queue,
                     int max_dets) {
  StreamFrame item;
  while (inference_queue.pop(item)) {
    item.detections = postprocess(item.output.get(), max_dets,
                                  item.frame->width, item.frame->height);
    postprocessor_queue.push(std::move(item));
  }
}

void run_draw(ThreadQueue<StreamFrame> &postprocessor_queue,
              ThreadQueue<StreamFrame> &encoder_queue) {
  StreamFrame item;
  while (postprocessor_queue.pop(item)) {
    draw_box(item.detections, item.frame);
    encoder_queue.push(std::move(item));
  }
}

bool encode_output_frame(OutputState &state, const AppConfig &config,
                         StreamFrame &item, std::atomic<bool> &pipeline_ok) {
  if (!state.output_opened) {
    std::string output_url =
        indexed_output_url(config.output_url, item.stream_index);
    if (!open_encoder(state.encoder, item.frame->width, item.frame->height,
                      (AVPixelFormat)item.frame->format, item.frame_rate)) {
      pipeline_ok = false;
      return false;
    }

    if (!open_server(state.server, output_url.c_str(),
                     state.encoder.codec_ctx)) {
      close_encoder(state.encoder);
      pipeline_ok = false;
      return false;
    }

    state.output_opened = true;
  }

  if (!serve_stream(state.encoder, state.server, item.frame)) {
    log_msg(LOG_ERROR, "main", "frame encode or publish failed");
    pipeline_ok = false;
    return false;
  }

  return true;
}

void encode_ready_frames(OutputState &state, const AppConfig &config,
                         std::atomic<bool> &pipeline_ok) {
  while (true) {
    auto it = state.pending_frames.find(state.next_frame_id);
    if (it == state.pending_frames.end())
      break;

    encode_output_frame(state, config, it->second, pipeline_ok);
    state.pending_frames.erase(it);
    state.next_frame_id++;
  }
}

void flush_output_state(OutputState &state, const AppConfig &config,
                        std::atomic<bool> &pipeline_ok) {
  std::lock_guard<std::mutex> lock(state.mutex);
  while (!state.pending_frames.empty()) {
    auto it = state.pending_frames.begin();
    if (it->first != state.next_frame_id) {
      log_msg(LOG_WARNING, "main",
              "encoding stream with missing frame_id expected=" +
                  std::to_string(state.next_frame_id) +
                  " got=" + std::to_string(it->first));
      state.next_frame_id = it->first;
    }

    encode_output_frame(state, config, it->second, pipeline_ok);
    state.pending_frames.erase(it);
    state.next_frame_id++;
  }

  if (state.output_opened) {
    flush_encoder(state.encoder, state.server);
    close_server(state.server);
    close_encoder(state.encoder);
    state.output_opened = false;
  }
}

void run_encode(const AppConfig &config,
                ThreadQueue<StreamFrame> &encoder_queue,
                std::vector<std::unique_ptr<OutputState>> &output_states,
                std::atomic<bool> &pipeline_ok) {
  StreamFrame item;
  while (encoder_queue.pop(item)) {
    if (item.stream_index < 0 ||
        item.stream_index >= (int)output_states.size()) {
      log_msg(LOG_ERROR, "main", "encoded frame has invalid stream index");
      pipeline_ok = false;
      continue;
    }

    OutputState &state = *output_states[item.stream_index];
    std::lock_guard<std::mutex> lock(state.mutex);

    auto [_, inserted] =
        state.pending_frames.emplace(item.frame_id, std::move(item));
    if (!inserted) {
      log_msg(LOG_ERROR, "main", "duplicate frame_id in encode queue");
      pipeline_ok = false;
      continue;
    }

    encode_ready_frames(state, config, pipeline_ok);
  }
}

bool run_stream(Engine &engine, const AppConfig &config, int batch_size) {
  int stream_count = (int)config.input_urls.size();
  int max_dets = (int)(engine.outputElementsPerFrame / 6);
  std::atomic<int> next_decode_stream{0};
  std::atomic<bool> pipeline_ok{true};

  ThreadQueue<StreamFrame> decoder_queue;
  ThreadQueue<StreamFrame> preprocessor_queue;
  ThreadQueue<StreamFrame> inference_queue;
  ThreadQueue<StreamFrame> postprocessor_queue;
  ThreadQueue<StreamFrame> encoder_queue;
  std::vector<std::unique_ptr<OutputState>> output_states;

  std::vector<std::thread> decoder_threads;
  std::vector<std::thread> preprocess_threads;
  std::vector<std::thread> postprocess_threads;
  std::vector<std::thread> draw_threads;
  std::vector<std::thread> encode_threads;

  decoder_threads.reserve(config.decoder_threads_count);
  preprocess_threads.reserve(config.preprocessor_threads_count);
  postprocess_threads.reserve(config.postprocessor_threads_count);
  draw_threads.reserve(config.drawer_threads_count);
  encode_threads.reserve(config.encoder_threads_count);
  output_states.reserve(stream_count);

  log_msg(LOG_INFO, "main",
          "max detections from model output: " + std::to_string(max_dets));

  for (int i = 0; i < config.decoder_threads_count; i++) {
    decoder_threads.emplace_back(
        run_decode, std::ref(config), std::ref(next_decode_stream),
        std::ref(decoder_queue), std::ref(pipeline_ok));
  }

  for (int i = 0; i < config.preprocessor_threads_count; i++) {
    preprocess_threads.emplace_back(run_preprocess, std::ref(decoder_queue),
                                    std::ref(preprocessor_queue));
  }

  std::thread infer_thread(run_infer, std::ref(engine),
                           std::ref(preprocessor_queue),
                           std::ref(inference_queue), batch_size);

  for (int i = 0; i < config.postprocessor_threads_count; i++) {
    postprocess_threads.emplace_back(run_postprocess, std::ref(inference_queue),
                                     std::ref(postprocessor_queue), max_dets);
  }

  for (int i = 0; i < config.drawer_threads_count; i++) {
    draw_threads.emplace_back(run_draw, std::ref(postprocessor_queue),
                              std::ref(encoder_queue));
  }

  for (int stream_index = 0; stream_index < stream_count; stream_index++) {
    output_states.push_back(std::make_unique<OutputState>());

    std::string output_url =
        indexed_output_url(config.output_url, stream_index);
    log_msg(LOG_INFO, "main",
            "stream " + std::to_string(stream_index) +
                " output: " + output_url);
  }

  for (int i = 0; i < config.encoder_threads_count; i++) {
    encode_threads.emplace_back(run_encode, std::ref(config),
                                std::ref(encoder_queue),
                                std::ref(output_states), std::ref(pipeline_ok));
  }

  for (auto &thread : decoder_threads)
    thread.join();
  decoder_queue.close();

  for (auto &thread : preprocess_threads)
    thread.join();
  preprocessor_queue.close();

  infer_thread.join();
  inference_queue.close();

  for (auto &thread : postprocess_threads)
    thread.join();
  postprocessor_queue.close();

  for (auto &thread : draw_threads)
    thread.join();

  encoder_queue.close();

  for (auto &thread : encode_threads)
    thread.join();

  for (auto &state : output_states)
    flush_output_state(*state, config, pipeline_ok);

  return pipeline_ok;
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
        build_engine_file(config.model_path.c_str(), config.engine_path.c_str(),
                          config.max_batch_size);
    log_msg(LOG_INFO, "main", "TensorRT engine build finished");
    return built ? 0 : -1;
  }

  int input_count = (int)config.input_urls.size();
  if (input_count == 0) {
    log_msg(LOG_ERROR, "main", "run mode requires at least one --input URL");
    return -1;
  }

  int hardware_threads =
      (int)std::thread::hardware_concurrency(); // 24 - 20 - 19
  int reserved_threads = hardware_threads >= 16  ? 4
                         : hardware_threads >= 8 ? 2
                                                 : 1;
  constexpr int inference_threads_count = 1;
  int max_threads =
      hardware_threads - reserved_threads - inference_threads_count;
  int stream_count = input_count;

  config.decoder_threads_count =
      std::max(config.decoder_threads_count, max_threads * 3 / 10);
  config.preprocessor_threads_count =
      std::max(config.preprocessor_threads_count, max_threads * 1 / 10);
  config.postprocessor_threads_count =
      std::max(config.postprocessor_threads_count, max_threads * 1 / 10);
  config.drawer_threads_count =
      std::max(config.drawer_threads_count, max_threads * 1 / 10);
  config.encoder_threads_count =
      std::max(config.encoder_threads_count, max_threads * 3 / 10);

  int batch_size = std::min(stream_count, config.max_batch_size);

  Engine engine;
  engine = load_engine(config.engine_path.c_str(), batch_size);
  log_msg(LOG_INFO, "main", "TensorRT engine load finished");

  if (!engine.context || !engine.inputBuffer || !engine.outputBuffer) {
    log_msg(LOG_ERROR, "main", "TensorRT engine is not ready");
    destroy_engine(engine);
    return -1;
  }

  bool stream_ok = run_stream(engine, config, batch_size);
  destroy_engine(engine);
  return stream_ok ? 0 : -1;
}
