#ifndef CUDAEXECUTOR_KERNEL_HPP_
#define CUDAEXECUTOR_KERNEL_HPP_

#include <string>
#include <vector>

#include "Options.hpp"
#include "ProgramArg.hpp"

#include <cuda.h>
#include <nvrtc.h>
#include <vector_types.h>

namespace cudaexecutor {

class Kernel {
  char *_ptx;
  dim3 _grid, _block;
  std::vector<std::string> _template_parameters;
  std::string _kernel_name, _name_expression, _log;
  nvrtcProgram *_prog; // maybe not a pointer?
  CUdevice _cuDevice;
  CUcontext _context;
  CUmodule _module;
  CUfunction _kernel;
  bool _compiled = false;

 public:
  Kernel(std::string function_name, nvrtcProgram *prog);
  ~Kernel();
  Kernel configure(dim3 grid, dim3 block);
  template <typename T> Kernel instantiate(T type);
  template <typename T, typename... TS> Kernel instantiate(T type, TS... types);
  Kernel launch(std::vector<ProgramArg> program_args);
  Kernel compile(Options options = Options());
  std::string log() const { return _log; }
};

} // namespace cudaexecutor

#endif // CUDAEXECUTOR_KERNEL_HPP_
