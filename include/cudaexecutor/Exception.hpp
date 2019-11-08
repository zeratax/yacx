#ifndef CUDAEXECUTOR_EXCEPTION_HPP_
#define CUDAEXECUTOR_EXCEPTION_HPP_

#include <cstdio>
#include <string>

#include <cuda.h>
#include <nvrtc.h>

namespace cudaexecutor {

class exception : public std::exception {
 protected:
  std::string _message;
  std::string _file;
  int _line;

 public:
  explicit exception(std::string message, std::string file = "", int line = 0)
      : _message{message}, _file{file}, _line{line} {};
  virtual const char *what() const throw() {
    std::string what = std::string("ERROR: ") + _message + "FILE: " + _file +
                       "LINE: " + std::to_string(_line);
    return what.c_str();
  }
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
