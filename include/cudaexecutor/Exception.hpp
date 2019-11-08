#ifndef CUDAEXECUTOR_EXCEPTION_HPP_
#define CUDAEXECUTOR_EXCEPTION_HPP_

#include <cstdio>
#include <string>

#include <cuda.h>
#include <nvrtc.h>

namespace cudaexecutor {

class exception : public std::exception {
 protected:
  std::string _what;
  std::string _message;
  std::string _file;
  int _line;

 public:
  explicit exception(std::string message, std::string file = "", int line = 0)
      : _message{message}, _file{file}, _line{line} {
    _what = std::string(_message + " in File: " + _file +
                        ", Line: " + std::to_string(_line));
  }
  void set_message(std::string message) {
    _message = message;
    _what = std::string(_message + " in File: " + _file +
                        ", Line: " + std::to_string(_line));
  }
  virtual const char *what() const throw() { return _message.c_str(); }
};

class cuda_exception : public exception {
 public:
  explicit cuda_exception(CUresult error, std::string file = "", int line = 0)
      : exception{"", file, line} {
    const char *cmessage = new char(64); // explicit destructor??
    cuGetErrorName(error, &cmessage);
    exception::_message = cmessage;
  }
};

class nvrtc_exception : public exception {
 public:
  explicit nvrtc_exception(nvrtcResult error, std::string file = "",
                           int line = 0)
      : exception{nvrtcGetErrorString(error), file, line} {};
};

} // namespace cudaexecutor

#endif // CUDAEXECUTOR_EXCEPTION_HPP_
