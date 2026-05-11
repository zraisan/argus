#pragma once

#include <chrono>
#include <string>

enum LogLevel { LOG_DEBUG, LOG_INFO, LOG_WARNING, LOG_ERROR, LOG_CRITICAL };

void log_msg(LogLevel level, const char *system, const std::string &message);
void log_duration(LogLevel level, const char *system, const char *name,
                  std::chrono::steady_clock::time_point start);
void log_av_error(LogLevel level, const char *system, const char *where,
                  int err);
