#include "../include/cudaexecutor/Program.hpp"
#include "../include/cudaexecutor/Exception.hpp"
#include "../include/cudaexecutor/Kernel.hpp"

#include <cuda.h>
#include <nvrtc.h>

#include <utility>

#define CUDA_SAFE_CALL(x)                                                      \
  do {                                                                         \
    CUresult result = x;                                                       \
    if (result != CUDA_SUCCESS) {                                              \
      throw cuda_exception(result);                                            \
    }                                                                          \
  } while (0)

using cudaexecutor::Program, cudaexecutor::ProgramArg, cudaexecutor::Kernel;

Program::Program(std::string kernel_string, Headers headers)
    : _kernel_string{std::move(kernel_string)}, _headers{std::move(headers)} {}

Program::~Program() {
  // exception in destruktor??
  // Destroy the program.
  NVRTC_SAFE_CALL(nvrtcDestroyProgram(_prog));
}

Kernel Program::kernel(const std::string &function_name) {
  _prog = new nvrtcProgram;                  // destructor?
  nvrtcCreateProgram(_prog,                  // prog
                     _kernel_string.c_str(), // buffer
                     function_name.c_str(),  // name
                     _headers.size(),        // numHeaders
                     _headers.names(),       // headers
                     _headers.content());    // includeNames
  return Kernel(function_name, _prog);
}
