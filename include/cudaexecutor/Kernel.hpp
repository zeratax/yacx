#ifndef CUDAEXECUTOR_KERNEL_HPP_
#define CUDAEXECUTOR_KERNEL_HPP_

#include <string>
#include <vector>

#include "Exception.hpp"
#include "Options.hpp"
#include "ProgramArg.hpp"

#include <cuda.h>
#include <nvrtc.h>
#include <vector_types.h>

#define NVRTC_SAFE_CALL(x)                                                     \
  do {                                                                         \
    nvrtcResult result = x;                                                    \
    if (result != NVRTC_SUCCESS) {                                             \
      throw nvrtc_exception(result, __FILE__, __LINE__);                       \
    }                                                                          \
  } while (0)

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

template <typename T> Kernel Kernel::instantiate(T type) {
  std::string type_name;

  NVRTC_SAFE_CALL(nvrtcGetTypeName<T>(&type_name));
  _compiled = false;
  _template_parameters.push_back(type_name);
}

template <typename T, typename... TS>
Kernel Kernel::instantiate(T type, TS... types) {
  std::string type_name;

  NVRTC_SAFE_CALL(nvrtcGetTypeName<T>(&type_name));
  _template_parameters.push_back(type_name);
  Kernel::instantiate(types...);
}

} // namespace cudaexecutor

#endif // CUDAEXECUTOR_KERNEL_HPP_
