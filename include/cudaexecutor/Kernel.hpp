#pragma once

#include "ProgramArg.hpp"
#include <cudaexecutor/Logger.hpp>

#include <cuda.h>
#include <nvrtc.h>
#include <vector>
#include <vector_types.h>

namespace cudaexecutor {

class Kernel {

  char *_ptx; // shared pointer?
  std::vector<std::string> _template_parameters;
  std::string _kernel_name, _name_expression;
  nvrtcProgram _prog;

  dim3 _grid, _block;
  CUdevice _cuDevice;
  CUcontext _context;
  CUmodule _module;
  CUfunction _kernel;

 public:
  explicit Kernel(char *_ptx, std::vector<std::string> template_parameters,
                  std::string kernel_name, std::string name_expression,
                  nvrtcProgram prog);
  Kernel &configure(dim3 grid, dim3 block);
  Kernel &launch(std::vector<ProgramArg> program_args);
};

} // namespace cudaexecutor
