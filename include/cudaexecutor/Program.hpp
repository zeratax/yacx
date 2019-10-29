#ifndef _PROGRAM_DEF_H_
#define _PROGRAM_DEF_H_

#include <string>
#include <vector>

#include "Headers.hpp"
#include "Kernel.hpp"
#include "util.hpp"

#include <cuda.h>
#include <nvrtc.h>

namespace cudaexecutor {

class Program {
  Headers headers;
  std::string kernel_string;

public:
  Program(std::kernel_string, Headers headers = Headers());
  Kernel kernel(std::string function_name);
};

class ProgramArg {
  void *hdata;
  CUdeviceptr ddata;
  bool output;

public:
  ProgramArg(const void *const data, bool output = false);
  void upload();
  void download();
};

} // namespace cudaexecutor

#endif // _PROGRAM_DEF_H_