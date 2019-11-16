#include "../include/cudaexecutor/Kernel.hpp"
#include "../include/cudaexecutor/Exception.hpp"
#include "../include/cudaexecutor/Headers.hpp"
#include "../include/cudaexecutor/util.hpp"

#include <memory>

#define NVRTC_SAFE_CALL(x)                                                     \
  do {                                                                         \
    nvrtcResult result = x;                                                    \
    if (result != NVRTC_SUCCESS) {                                             \
      throw nvrtc_exception(result, __FILE__, __LINE__);                       \
    }                                                                          \
  } while (0)
#define CUDA_SAFE_CALL(x)                                                      \
  do {                                                                         \
    CUresult result = x;                                                       \
    if (result != CUDA_SUCCESS) {                                              \
      throw cuda_exception(result, std::string(__FILE__), __LINE__);           \
    }                                                                          \
  } while (0)

using cudaexecutor::Kernel, cudaexecutor::Options, cudaexecutor::Headers,
    cudaexecutor::ProgramArg, cudaexecutor::to_comma_separated;

Kernel::Kernel(std::string kernel_name, nvrtcProgram *prog)
    : _kernel_name{kernel_name}, _prog{prog} {
  // optional parameter to take device, check if already initialized??
  CUDA_SAFE_CALL(cuInit(0));
  CUDA_SAFE_CALL(cuDeviceGet(&_cuDevice, 0));
  CUDA_SAFE_CALL(cuCtxCreate(&_context, 0, _cuDevice));
  CUDA_SAFE_CALL(cuModuleLoadDataEx(&_module, _ptx, 0, 0, 0));
}

Kernel::~Kernel() {
  // Exceptions in deconstructor usually a bad idea??
  // Release resources.
  CUDA_SAFE_CALL(cuModuleUnload(_module));
  CUDA_SAFE_CALL(cuCtxDestroy(_context));
}

Kernel Kernel::configure(dim3 grid, dim3 block) {
  _grid = grid;
  _block = block;
  return *this;
}

Kernel Kernel::launch(std::vector<ProgramArg> args) {
  if (!_compiled)
    throw std::invalid_argument("kernel needs to be compiled to be launched");

  void *kernel_args[args.size()];
  int i{0};
  for (auto &arg : args) {
    arg.upload();
    kernel_args[i++] = arg.content();
  }

  // lowered name
  const char *name;
  NVRTC_SAFE_CALL(nvrtcGetLoweredName(*_prog, _name_expression.c_str(), &name));

  CUDA_SAFE_CALL(cuModuleGetFunction(&_kernel, _module, name));

  // launch the kernel

  std::cout << "\nlaunching " << name << " (" << _name_expression << ")"
            << std::endl;
  CUDA_SAFE_CALL(cuLaunchKernel(_kernel, // function from kernel
                                _grid.x, _grid.y, _grid.z,    // grid dim
                                _block.x, _block.y, _block.z, // block dim
                                0, NULL,          // shared mem and stream
                                kernel_args, 0)); // arguments
  CUDA_SAFE_CALL(cuCtxSynchronize());

  // download results to host
  for (auto &arg : args)
    arg.download();

  return *this;
}

Kernel Kernel::compile(Options options) {
  _name_expression =
      _kernel_name + "<" + to_comma_separated(_template_parameters) + ">";
  NVRTC_SAFE_CALL(nvrtcAddNameExpression(*_prog, _name_expression.c_str()));

  nvrtcResult compileResult =
      nvrtcCompileProgram(*_prog, options.numOptions(), options.options());

  size_t logSize;
  NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(*_prog, &logSize));
  std::unique_ptr<char[]> clog = std::make_unique<char[]>(logSize);
  NVRTC_SAFE_CALL(nvrtcGetProgramLog(*_prog, clog.get()));
  _log = clog.get();

  if (compileResult != NVRTC_SUCCESS)
    throw nvrtc_exception(compileResult);

  size_t ptxSize;
  NVRTC_SAFE_CALL(nvrtcGetPTXSize(*_prog, &ptxSize));
  _ptx = new char[ptxSize];
  NVRTC_SAFE_CALL(nvrtcGetPTX(*_prog, _ptx));

  _compiled = true;
  return *this;
}
