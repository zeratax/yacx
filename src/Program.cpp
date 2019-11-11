#include "../include/cudaexecutor/Program.hpp"
#include "../include/cudaexecutor/Exception.hpp"
#include "../include/cudaexecutor/Kernel.hpp"

#include <cuda.h>
#include <nvrtc.h>

#define CUDA_SAFE_CALL(x)                                                      \
  do {                                                                         \
    CUresult result = x;                                                       \
    if (result != CUDA_SUCCESS) {                                              \
      throw cuda_exception(result);                                            \
    }                                                                          \
  } while (0)

using cudaexecutor::Program, cudaexecutor::ProgramArg, cudaexecutor::Kernel;

Program::Program(std::string kernel_string, Headers headers)
    : _kernel_string{kernel_string}, _headers{headers} {}

Kernel Program::kernel(std::string function_name) {
  _prog = new nvrtcProgram; // destruktor?
  nvrtcCreateProgram(_prog,                   // prog
                     _kernel_string.c_str(), // buffer
                     function_name.c_str(),  // name
                     _headers.size(),        // numHeaders
                     _headers.names(),       // headers
                     _headers.content());    // includeNames
  return Kernel(function_name, _prog);
}
