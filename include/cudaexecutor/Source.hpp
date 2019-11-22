#pragma once

#include <string>
#include <vector>

#include "Headers.hpp"
#include "Program.hpp"
#include "util.hpp"

#include <cuda.h>
#include <nvrtc.h>

namespace cudaexecutor {

class Source {
  std::string _kernel_string;
  Headers _headers;
  nvrtcProgram *_prog = nullptr;

 public:
  explicit Source(std::string kernel_string, Headers headers = Headers());
  ~Source();
  Program program(const std::string &function_name);
};

} // namespace cudaexecutor
