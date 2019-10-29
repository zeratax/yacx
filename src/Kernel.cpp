#include "Kernel.hpp"
#include "Exception.hpp"
#include "Headers.hpp"
#include "Options.hpp"
#include "Program.hpp"

#define NVRTC_SAFE_CALL(x)                                                     \
  do {                                                                         \
    nvrtcResult result = x;                                                    \
    if (result != NVRTC_SUCCESS) {                                             \
      throw nvrtc_exception(result)                                            \
    }                                                                          \
  } while (0)

using cudaexecutor::Kernel, cudaexecutor::Options, cudaexecutor::Headers,
    cudaexecutor::Program;

Kernel::Kernel(std::kernel_string, Headers headers = Headers())
    : this->kernel_string{kernel_string},
    this->headers{headers} {}

Program Kernel::compile(Options options = Options()) {
  nvrtcProgram prog = new nvrtcProgram;
  nvrtcCreateProgram(&prog,                             // prog
                     this->kernel_string.c_str(),       // buffer
                     this->kernel_name.c_str(),         // name
                     this->headers.size(),              // numHeaders
                     this->headers.names(),             // headers
                     this->headers.content());          // includeNames
  nvrtcResult compileResult = nvrtcCompileProgram(prog, // prog
                                                  options.size(),     //
                                                  options.content()); // options

  // Obtain compilation log from the program.
  size_t logSize;
  NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize));
  char *clog = new char[logSize];
  NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, clog));
  this->log = clog;

  if (compileResult != NVRTC_SUCCESS)
    throw nvrtc_exception(result)

        size_t ptxSize;
  NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));
  char *ptx = new char[ptxSize];
  NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptx));

  return new Program(ptx, prog);
}
