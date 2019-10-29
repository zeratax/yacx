#ifndef _PROGRAM_DEF_H_
#define _PROGRAM_DEF_H_

#include <string>
#include <vector>

#include "Kernel.hpp"
#include "util.hpp"

#include <cuda.h>
#include <nvrtc.h>

namespace cudaexecutor {

template <int DIM_GRID, int DIM_BLOCK> typedef struct Grid {
  dim3 blocks(DIM_GRID);
  dim3 threads(DIM_GRID);
};

class Program {
  char *ptx;
  char **args;
  nvrtcProgram prog;
  std::string kernel_name;
  std::string name_expression;
  Grid grid;
  CUdevice cuDevice;
  CUcontext context;
  CUmodule module;
  CUfunction kernel;

public:
  Program(char *ptx, nvrtcProgram prog);
  Program kernel(std::string kernel_name);
  Program configure(Grid grid);
  Program instantiate(std::string... types);
  Program launch(std::vector<ProgramArg> program_args);
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