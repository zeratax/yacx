#ifndef CUDAEXECUTOR_EXCEPTION_HPP_
#define CUDAEXECUTOR_EXCEPTION_HPP_

#include <cstdio>
#include <string>
#include <utility>

#include <cuda.h>
#include <nvrtc.h>
#include <memory>

namespace cudaexecutor {
// TODO(zeratax): Properly show line and file
class exception : public std::exception {
 protected:
  std::string _what;
  std::string _message;
  std::string _file;
  int _line;

  void set_message(std::string message) {
    _message = std::move(message);
    _what = std::string(_message + " in File: " + _file +
                        ", Line: " + std::to_string(_line));
  }

 public:
  explicit exception(std::string message, std::string file = "", int line = 0);
  [[nodiscard]] const char *what() const noexcept override {
    return _what.c_str();
  }
};

exception::exception(std::string message, std::string file, int line)
    : _message{std::move(message)}, _file{std::move(file)}, _line{line} {
  _what = std::string(_message + " in File: " + _file +
                      ", Line: " + std::to_string(_line));
}

class cuda_exception : public exception {
 public:
  explicit cuda_exception(CUresult error, std::string file = "", int line = 0);
};

cuda_exception::cuda_exception(CUresult error, std::string file, int line)
    : exception{"", std::move(file), line} {
//  const char *cmessage = new char[64];
//  cuGetErrorName(error, &cmessage);
//  set_message(cmessage);
   std::unique_ptr<const char*> cmessage = std::make_unique<const char*>(64);
   cuGetErrorName(error, cmessage.get());
   set_message(*cmessage);
}

class nvrtc_exception : public exception {
 public:
  explicit nvrtc_exception(nvrtcResult error, std::string file = "",
                           int line = 0)
      : exception{nvrtcGetErrorString(error), file, line} {};
};

} // namespace cudaexecutor

#endif // CUDAEXECUTOR_EXCEPTION_HPP_
