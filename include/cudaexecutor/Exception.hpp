#ifndef _EXCEPTION_H_
#define _EXCEPTION_H_

#include <cstdio>
#include <string>

#include <cuda.h>
#include <nvrtc.h>

namespace cudaexecutor {

class cuda_exception : public std::exception {
  std::string message;
  virtual const char *what() const throw() { return message; }

public:
  cuda_exception(int error) {
    const char *cmessage = new char *; // explicit destructor??
    cuGetErrorName(error, &cmessage);
    this.message = cmessage;
  }
};

class nvrtc_exception : public std::exception {
  std::string message;
  virtual const char *what() const throw() { return message; }

public:
  cuda_exception(int error) { this.message = nvrtcGetErrorString(error); }
};

} // namespace cudaexecutor

#endif // _EXCEPTION_H_