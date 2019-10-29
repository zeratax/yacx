#ifndef _KERNEL_H_
#define _KERNEL_H_

#include <string>
#include <vector>

#include "Options.hpp"
#include "Program.hpp"

#include <cuda.h>
#include <nvrtc.h>

namespace cudaexecutor {

template <int DIM_GRID, int DIM_BLOCK> typedef struct Grid {
  dim3 blocks(DIM_GRID);
  dim3 threads(DIM_GRID);
};

class Kernel {
  char *ptx;
  char **args;
  std::string kernel_name;
  std::string function_name;
  std::string name_expression;
  std::string log;
  nvrtcProgram prog;
  CUdevice cuDevice;
  CUcontext context;
  CUmodule module;
  CUfunction kernel;
  Grid grid;

public:
  Kernel(std::string kernel_string, std::string function_name,
         Headers headers = Headers());
  Kernel configure(Grid grid);
  Kernel instantiate(std::string... types);
  Kernel launch(std::vector<ProgramArg> program_args);
  Kernel compile(Options options = Options());
  std::string log() { return log; }
};

} // namespace cudaexecutor

#endif // _KERNEL_H_
