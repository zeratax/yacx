#ifndef _KERNEL_H_
#define _KERNEL_H_

#include <string>
#include <vector>

#include "Options.hpp"
#include "Program.hpp"

#include <cuda.h>
#include <nvrtc.h>

namespace cudaexecutor {

class Kernel {
  char *_ptx;
  char **args;
  std::string _kernel_name;
  std::string _function_name;
  std::string _name_expression;
  std::string _log;
  nvrtcProgram _prog;
  CUdevice _cuDevice;
  CUcontext _context;
  CUmodule _module;
  CUfunction _kernel;

 public:
  Kernel(std::string kernel_string, std::string function_name,
         Headers headers = Headers());
  Kernel configure(dim3 grid, dim3 block);
  template <typename T> Kernel instantiate(T types);
  template <typename T, typename... Args>
  Kernel instantiate(T a, T b, Args... types);
  Kernel launch(std::vector<ProgramArg> program_args);
  Kernel compile(Options options = Options());
  std::string log() { return _log; }
};

} // namespace cudaexecutor

#endif // _KERNEL_H_
