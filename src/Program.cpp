#include "Program.hpp"

#define CUDA_SAFE_CALL(x)                                                      \
  do {                                                                         \
    CUresult result = x;                                                       \
    if (result != CUDA_SUCCESS) {                                              \
      throw cuda_exception(result)                                             \
    }                                                                          \
  } while (0)
#define NVRTC_SAFE_CALL(x)                                                     \
  do {                                                                         \
    nvrtcResult result = x;                                                    \
    if (result != NVRTC_SUCCESS) {                                             \
      throw nvrtc_exception(result)                                            \
    }                                                                          \
  } while (0)

using cudaexecutor::Program, cudaexecutor::ProgramArg;

Program::Program(char *ptx, nvrtcProgram prog)
    : this->ptx{ptx},
    this->prog{prog} {
  CUDA_SAFE_CALL(cuInit(0));
  CUDA_SAFE_CALL(cuDeviceGet(&this->cuDevice, 0));
  CUDA_SAFE_CALL(cuCtxCreate(&this->context, 0, this->cuDevice));
  CUDA_SAFE_CALL(cuModuleLoadDataEx(&this->module, this->ptx, 0, 0, 0));
}

~Program::Program() {
  // Release resources.
  CUDA_SAFE_CALL(cuModuleUnload(this->module));
  CUDA_SAFE_CALL(cuCtxDestroy(this->context));

  // Destroy the program.
  NVRTC_SAFE_CALL(nvrtcDestroyProgram(&this->prog));
}

Program Program::kernel(std::string kernel_name) {
  this->kernel_name = kernel_name;
  return *this;
}

Program Program::configure(dim3 grid, dim3 block)) {
  this->grid = grid;
  this->block = block;
  return *this;
}

Program Program::launch(std::vector<ProgramArg> program_args) {
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
