#ifndef _KERNEL_H_
#define _KERNEL_H_

#include <string>
#include <vector>

#include "Headers.hpp"
#include "Options.hpp"
#include "Program.hpp"

#include <nvrtc.h>

namespace cudaexecutor {

class Kernel {
  Headers _headers;
  std::string kernel_string;
  std::string log;

public:
  Kernel(std::kernel_string, Headers headers = Headers());
  Kernel instantiate(std::string... types);
  Program compile(Options options = Options());
  std::string log() { return log; }
};

} // namespace cudaexecutor

#endif // _KERNEL_H_
