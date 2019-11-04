#ifndef _EXCEPTION_H_
#define _EXCEPTION_H_

#include <cstdio>
#include <string>

#include <cuda.h>
#include <nvrtc.h>

namespace cudaexecutor {

class cuda_exception : public std::exception {
  std::string _message;

public:
  explicit cuda_exception(int error) {
    const char *cmessage = new char(64); // explicit destructor??
    cuGetErrorName(static_cast<CUresult>(error), &cmessage);
    _message = cmessage;
  }
  virtual const char *what() const throw() { return _message.c_str(); }
};

class nvrtc_exception : public std::exception {
  std::string _message;

public:
  explicit nvrtc_exception(int error) {
    _message = nvrtcGetErrorString(static_cast<nvrtcResult>(error));
  }
  virtual const char *what() const throw() { return _message.c_str(); }
};

} // namespace cudaexecutor

#endif // _EXCEPTION_H_
