#ifndef CUDAEXECUTOR_PROGRAM_HPP_
#define CUDAEXECUTOR_PROGRAM_HPP_

#include <string>
#include <vector>

#include "Headers.hpp"
#include "Kernel.hpp"
#include "util.hpp"

#include <cuda.h>
#include <nvrtc.h>

namespace cudaexecutor {

class Program {
  Headers _headers;
  nvrtcProgram *_prog;
  std::string _kernel_string;

 public:
  explicit Program(std::string kernel_string, Headers headers = Headers());
  Kernel kernel(std::string function_name);
};

} // namespace cudaexecutor

#endif // CUDAEXECUTOR_PROGRAM_HPP_
