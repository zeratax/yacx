#include "../include/cudaexecutor/Kernel.hpp"
#include "../include/cudaexecutor/Exception.hpp"
#include "../include/cudaexecutor/Headers.hpp"
#include "../include/cudaexecutor/Options.hpp"
#include "../include/cudaexecutor/Program.hpp"

#define NVRTC_SAFE_CALL(x)                                                     \
  do {                                                                         \
    nvrtcResult result = x;                                                    \
    if (result != NVRTC_SUCCESS) {                                             \
      throw nvrtc_exception(result);                                           \
    }                                                                          \
  } while (0)
#define CUDA_SAFE_CALL(x)                                                      \
  do {                                                                         \
    CUresult result = x;                                                       \
    if (result != CUDA_SUCCESS) {                                              \
      throw cuda_exception(result);                                            \
    }                                                                          \
  } while (0)

using cudaexecutor::Kernel, cudaexecutor::Options, cudaexecutor::Headers,
    cudaexecutor::Program;

Kernel::Kernel(std::function_name, nvrtcProgram prog)
    : _function_name{function_name}, _prog{prog} {
  CUDA_SAFE_CALL(cuInit(0));
  CUDA_SAFE_CALL(cuDeviceGet(&_cuDevice, 0));
  CUDA_SAFE_CALL(cuCtxCreate(&_context, 0, _cuDevice));
  CUDA_SAFE_CALL(cuModuleLoadDataEx(&_module, _ptx, 0, 0, 0));
}

~Kernel::Kernel() {
  // Release resources.
  CUDA_SAFE_CALL(cuModuleUnload(_module));
  CUDA_SAFE_CALL(cuCtxDestroy(_context));

  // Destroy the program.
  NVRTC_SAFE_CALL(nvrtcDestroyProgram(_prog));
}

Kernel Kernel::configure(dim3 grid, dim3 block)) {
  _grid = grid;
  _block = block;
  return *this;
}

Kernel Kernel::launch(std::vector<ProgramArg> program_args) {
  // upload args to device
  for (auto &arg : program_args)
    arg.upload();

  // extract the lowered name for corresponding __global__ function,
  const char *name;
  NVRTC_SAFE_CALL(
      nvrtcGetLoweredName(_prog,
                          _name_expresion.c_str(), // name expression
                          &name));                 // lowered name

  CUDA_SAFE_CALL(cuModuleGetFunction(&_kernel, _module, name));

  // launch the kernel
  std::cout << "\nlaunching " << name << " (" << _name_expresion << ")"
            << std::endl;
  CUDA_SAFE_CALL(cuLaunchKernel(kernel, // function from kernel
                                _grid.x, _grid.y, _grid.z,    // grid dim
                                _block.x, _block.y, _block.z, // block dim
                                0, NULL,    // shared mem and stream
                                _args, 0)); // arguments
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
  NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(_prog, &logSize));
  char *clog = new char[logSize];
  NVRTC_SAFE_CALL(nvrtcGetProgramLog(_prog, clog));
  this->log = clog;

  if (compileResult != NVRTC_SUCCESS)
    throw nvrtc_exception(result);

  size_t ptxSize;
  NVRTC_SAFE_CALL(nvrtcGetPTXSize(_prog, &ptxSize));
  char *ptx = new char[ptxSize];
  NVRTC_SAFE_CALL(nvrtcGetPTX(_prog, ptx));

  return *this;
}
