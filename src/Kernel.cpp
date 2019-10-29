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
#define CUDA_SAFE_CALL(x)                                                      \
  do {                                                                         \
    CUresult result = x;                                                       \
    if (result != CUDA_SUCCESS) {                                              \
      throw cuda_exception(result)                                             \
    }                                                                          \
  } while (0)

using cudaexecutor::Kernel, cudaexecutor::Options, cudaexecutor::Headers,
    cudaexecutor::Program;

Kernel::Kernel(std::function_name, nvrtcProgram prog)
    : this->function_name{function_name},
    this->prog{prog} {
  CUDA_SAFE_CALL(cuInit(0));
  CUDA_SAFE_CALL(cuDeviceGet(&this->cuDevice, 0));
  CUDA_SAFE_CALL(cuCtxCreate(&this->context, 0, this->cuDevice));
  CUDA_SAFE_CALL(cuModuleLoadDataEx(&this->module, this->ptx, 0, 0, 0));
}

~Kernel::Kernel() {
  // Release resources.
  CUDA_SAFE_CALL(cuModuleUnload(this->module));
  CUDA_SAFE_CALL(cuCtxDestroy(this->context));

  // Destroy the program.
  NVRTC_SAFE_CALL(nvrtcDestroyProgram(&this->prog));
}

Kernel Kernel::configure(dim3 grid, dim3 block)) {
  this->grid = grid;
  this->block = block;
  return *this;
}

Kernel Kernel::launch(std::vector<ProgramArg> program_args) {
  // upload args to device
  for (auto &arg : program_args)
    arg.upload();

  // extract the lowered name for corresponding __global__ function,
  const char *name;
  NVRTC_SAFE_CALL(
      nvrtcGetLoweredName(this->prog,
                          this->name_expresion.c_str(), // name expression
                          &name));                      // lowered name

  CUDA_SAFE_CALL(cuModuleGetFunction(&this->kernel, this->module, name));

  // launch the kernel
  std::cout << "\nlaunching " << name << " (" << this->name_expresion << ")"
            << std::endl;
  CUDA_SAFE_CALL(
      cuLaunchKernel(kernel, // function from kernel
                     this->grid.x, this->grid.y, this->grid.z,    // grid dim
                     this->block.x, this->block.y, this->block.z, // block dim
                     0, NULL,         // shared mem and stream
                     this->args, 0)); // arguments
  CUDA_SAFE_CALL(cuCtxSynchronize());

  // download results to host
  for (auto &arg : program_args)
    arg.download();

  return *this;
}

Kernel Kernel::compile(Options options = Options()) {
  nvrtcResult compileResult = nvrtcCompileProgram(this->prog,         // progam
                                                  options.size(),     //
                                                  options.content()); // options

  // Obtain compilation log from the program.
  size_t logSize;
  NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(this->prog, &logSize));
  char *clog = new char[logSize];
  NVRTC_SAFE_CALL(nvrtcGetProgramLog(this->prog, clog));
  this->log = clog;

  if (compileResult != NVRTC_SUCCESS)
    throw nvrtc_exception(result)

        size_t ptxSize;
  NVRTC_SAFE_CALL(nvrtcGetPTXSize(this->prog, &ptxSize));
  char *ptx = new char[ptxSize];
  NVRTC_SAFE_CALL(nvrtcGetPTX(this->prog, ptx));

  return *this;
}
