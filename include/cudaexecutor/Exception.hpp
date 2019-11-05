#ifndef CUDAEXECUTOR_EXCEPTION_HPP_
#define CUDAEXECUTOR_EXCEPTION_HPP_

#include <cstdio>
#include <string>

#include <cuda.h>
#include <nvrtc.h>

namespace cudaexecutor {

class cudaexec_exception : public std::exception {
  std::string _message;
  std::string _file;
  std::string _line;

public:
  explicit cudaexec_exception(std::string message, std::string file = "",
                              std::string line = "")
      : _message{message}, _file{file}, _line{line} {};
  virtual const char *what() const throw() { return _message.c_str(); }
};

class cuda_exception : public cudaexec_exception {
public:
  explicit cuda_exception(int error, std::string file = "",
                          std::string line = "") {
    const char *cmessage = new char(64); // explicit destructor??
    cuGetErrorName(static_cast<CUresult>(error), &cmessage);
    cudaexec_exception{cmessage, file, line};
  }
};

class nvrtc_exception : public cudaexec_exception {
public:
  explicit nvrtc_exception(int error, std::string file = "",
                           std::string line = "")
      : cudaexec_exception{nvrtcGetErrorString(static_cast<nvrtcResult>(error)),
                           file, line} {};
};

} // namespace cudaexecutor

#endif // CUDAEXECUTOR__EXCEPTION_HPP_
