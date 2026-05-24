#pragma once

#include <cstddef>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <string>
#include <utility>

extern "C" {
#include <libavutil/pixfmt.h>
#include <libavutil/rational.h>
}

enum class StreamType {
  Unknown,
  File,
  Http,
  Hls,
  Mp4,
  MpegTs,
  Matroska,
  Rtsp,
  Webrtc
};

std::string av_rational_string(AVRational rational);
std::string av_pixel_format_name(AVPixelFormat pixel_format);
StreamType stream_type_from_url(const std::string &url);

template <typename T> class ThreadQueue {
public:
  explicit ThreadQueue(std::size_t capacity = 4) : capacity_(capacity) {}

  bool push(T item) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      not_full_.wait(lock,
                     [&] { return closed_ || queue_.size() < capacity_; });
      if (closed_)
        return false;
      queue_.push(std::move(item));
    }
    not_empty_.notify_one();
    return true;
  }

  bool pop(T &item) {
    std::unique_lock<std::mutex> lock(mutex_);
    not_empty_.wait(lock, [&] { return closed_ || !queue_.empty(); });

    if (queue_.empty())
      return false;

    item = std::move(queue_.front());
    queue_.pop();
    not_full_.notify_one();
    return true;
  }

  bool try_pop(T &item) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty())
      return false;

    item = std::move(queue_.front());
    queue_.pop();
    not_full_.notify_one();
    return true;
  }

  bool closed_and_empty() {
    std::lock_guard<std::mutex> lock(mutex_);
    return closed_ && queue_.empty();
  }

  void close() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      closed_ = true;
    }
    not_empty_.notify_all();
    not_full_.notify_all();
  }

private:
  std::queue<T> queue_;
  std::size_t capacity_;
  std::mutex mutex_;
  std::condition_variable not_empty_;
  std::condition_variable not_full_;
  bool closed_ = false;
};
