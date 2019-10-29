#include "Program.hpp"

#define CUDA_SAFE_CALL(x)                                                      \
  do {                                                                         \
    CUresult result = x;                                                       \
    if (result != CUDA_SUCCESS) {                                              \
      throw cuda_exception(result)                                             \
    }                                                                          \
  } while (0)

using cudaexecutor::Program, cudaexecutor::ProgramArg;

Program::Program(std::string kernel_string, Headers headers = Headers())
    : this->kernel_string{kernel_string},
    this->headers{headers} {}

Kernel Program::kernel(std::string function_name) {
  nvrtcProg prog = new nvrtcProg;
  nvrtcCreateProgram(&this->prog,                 // prog
                     this->kernel_string.c_str(), // buffer
                     this->kernel_name.c_str(),   // name
                     this->headers.size(),        // numHeaders
                     this->headers.names(),       // headers
                     this->headers.content());    // includeNames
  return new Kernel(function_name, prog);
}

ProgramArg::ProgramArg(const void *const data, bool output = false)
    : this->hdata{data},
    this->output{output} {}

void ProgramArg::upload() {
  CUDA_SAFE_CALL(cuMemAlloc(&this->ddata, sizeof(this->hdata)));
  CUDA_SAFE_CALL(cuMemcpyHtoD(this->ddata, &this->hdata, sizeof(this->hdata)));
}

void ProgramArg::download() {
  if (this->output)
    CUDA_SAFE_CALL(
        cuMemcpyDtoH(&this->hdata, this->ddata, sizeof(this->hdata)));
  CUDA_SAFE_CALL(cuMemFree(this->ddata));
}
