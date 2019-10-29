#include "../include/cudaexecutor/Program.hpp"

#define CUDA_SAFE_CALL(x)                                                      \
  do {                                                                         \
    CUresult result = x;                                                       \
    if (result != CUDA_SUCCESS) {                                              \
      throw cuda_exception(result);                                            \
    }                                                                          \
  } while (0)

using cudaexecutor::Program, cudaexecutor::ProgramArg;

Program::Program(std::string kernel_string, Headers headers = Headers())
    : _kernel_string{kernel_string}, _headers{headers} {}

Kernel Program::kernel(std::string function_name) {
  nvrtcProg prog = new nvrtcProg;
  nvrtcCreateProgram(&_prog,                 // prog
                     _kernel_string.c_str(), // buffer
                     _kernel_name.c_str(),   // name
                     _headers.size(),        // numHeaders
                     _headers.names(),       // headers
                     _headers.content());    // includeNames
  return new Kernel(function_name, prog);
}

ProgramArg::ProgramArg(const void *const data, bool output = false)
    : _hdata{data}, _output{output} {}

void ProgramArg::upload() {
  CUDA_SAFE_CALL(cuMemAlloc(&_ddata, sizeof(_hdata)));
  CUDA_SAFE_CALL(cuMemcpyHtoD(_ddata, &_hdata, sizeof(_hdata)));
}

void ProgramArg::download() {
  if (_output)
    CUDA_SAFE_CALL(cuMemcpyDtoH(&_hdata, _ddata, sizeof(_hdata)));
  CUDA_SAFE_CALL(cuMemFree(_ddata));
}
