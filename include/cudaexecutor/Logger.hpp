#pragma once

#include <iostream>
#include <map>
#include <sstream>

namespace cudaexecutor {

enum class loglevel { NONE, ERROR, WARNING, INFO, DEBUG, DEBUG1 };

class logIt {
  loglevel _level;
  loglevel _current_level;

 public:
  logIt(loglevel level, loglevel current_level, const char *file,
        const int line)
      : _level{level}, _current_level{current_level} {
    _buffer << "LOGGER:[" << file << ":" << line << "]: ";
  }

  template <typename T> logIt &operator<<(T const &value) {
    _buffer << value;
    return *this;
  }

  ~logIt() {
    _buffer << std::endl;
    if (static_cast<int>(_level) <= static_cast<int>(_current_level))
      std::cerr << _buffer.str();
  }

 private:
  std::ostringstream _buffer;
};

#ifdef current_log_level
#define logger(level)                                                          \
  if (static_cast<int>(level) > static_cast<int>(current_log_level))           \
    ;                                                                          \
  else                                                                         \
    cudaexecutor::logIt(level, current_log_level, __FILE__, __LINE__)
#else
#define logger(level) logIt(level, loglevel::ERROR, __FILE__, __LINE__)
#endif

} // namespace cudaexecutor

