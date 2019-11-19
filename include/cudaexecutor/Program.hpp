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
  std::string _kernel_string;
  Headers _headers;
  nvrtcProgram *_prog = nullptr;

 public:
  explicit Program(std::string kernel_string, Headers headers = Headers());
  ~Program();
  Kernel kernel(const std::string &function_name);
};

} // namespace cudaexecutor

#endif // CUDAEXECUTOR_PROGRAM_HPP_
