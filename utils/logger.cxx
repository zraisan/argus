#include "logger.hxx"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>

extern "C" {
#include <libavutil/error.h>
}

namespace {

const auto start_time = std::chrono::steady_clock::now();

const char *log_level_name(LogLevel level) {
  switch (level) {
  case LOG_DEBUG:
    return "DEBUG";
  case LOG_INFO:
    return "INFO";
  case LOG_WARNING:
    return "WARN";
  case LOG_ERROR:
    return "ERROR";
  case LOG_CRITICAL:
    return "CRIT";
  }
  return "INFO";
}

double seconds_since(std::chrono::steady_clock::time_point start) {
  std::chrono::duration<double> elapsed =
      std::chrono::steady_clock::now() - start;
  return elapsed.count();
}

std::string duration_message(const char *name, double seconds) {
  std::ostringstream out;
  out << name << " finished in " << std::fixed << std::setprecision(3)
      << seconds << "s";
  return out.str();
}

} // namespace

void log_msg(LogLevel level, const char *system, const std::string &message) {
  std::ostringstream out;
  out << "[" << std::fixed << std::setprecision(3) << std::setw(9)
      << std::setfill('0') << seconds_since(start_time) << "s]"
      << " [" << log_level_name(level) << "]"
      << " [" << system << "] " << message;
  std::cout << out.str() << std::endl;
}

void log_duration(LogLevel level, const char *system, const char *name,
                  std::chrono::steady_clock::time_point start) {
  log_msg(level, system, duration_message(name, seconds_since(start)));
}

void log_av_error(LogLevel level, const char *system, const char *where,
                  int err) {
  char buf[AV_ERROR_MAX_STRING_SIZE] = {};
  av_strerror(err, buf, sizeof(buf));

  std::ostringstream msg;
  msg << where << ": " << buf << " (" << err << ")";
  log_msg(level, system, msg.str());
}
