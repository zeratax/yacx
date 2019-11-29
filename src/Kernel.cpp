#include "cudaexecutor/Kernel.hpp"
#include "cudaexecutor/Exception.hpp"
#include <utility>

using cudaexecutor::Kernel, cudaexecutor::loglevel;

Kernel::Kernel(std::shared_ptr<char[]> ptx, const char *demangled_name)
    : _ptx{std::move(ptx)}, _demangled_name{demangled_name} {
  logger(loglevel::DEBUG) << "created templated Kernel " << _demangled_name;
}

Kernel &Kernel::configure(dim3 grid, dim3 block) {
  logger(loglevel::DEBUG) << "configuring Kernel with grid: " << grid.x << ", "
                          << grid.y << ", " << grid.z << " and block "
                          << block.x << ", " << block.y << ", " << block.z;
  _grid = grid;
  _block = block;
  return *this;
}

Kernel &Kernel::launch(std::vector<ProgramArg> args) {
  logger(loglevel::DEBUG) << "creating context and loading module";

  // check if device already initialised
  CUDA_SAFE_CALL(cuDeviceGet(&_cuDevice, 0));

  CUDA_SAFE_CALL(cuCtxCreate(&_context, 0, _cuDevice));
  logger(loglevel::DEBUG1) << _ptx.get();
  CUDA_SAFE_CALL(cuModuleLoadDataEx(&_module, _ptx.get(), 0, nullptr, nullptr));

  logger(loglevel::DEBUG) << "uploading arguments";
  const void *kernel_args[args.size()];
  int i{0};
  for (auto &arg : args) {
    arg.upload();
    kernel_args[i++] = arg.content();
  }
  logger(loglevel::DEBUG) << "getting function for " << _demangled_name;
  // const char *name;
  CUDA_SAFE_CALL(cuModuleGetFunction(&_kernel, _module, _demangled_name));

  // launch the program

  logger(loglevel::INFO) << "launching " << _demangled_name;
  CUDA_SAFE_CALL(cuLaunchKernel(_kernel, // function from program
                                _grid.x, _grid.y, _grid.z,    // grid dim
                                _block.x, _block.y, _block.z, // block dim
                                0, nullptr, // shared mem and stream
                                const_cast<void **>(kernel_args),
                                nullptr)); // arguments
  CUDA_SAFE_CALL(cuCtxSynchronize());
  logger(loglevel::INFO) << "done!";

  // download results to host
  logger(loglevel::DEBUG) << "downloading arguments";
  for (auto &arg : args)
    arg.download();

  logger(loglevel::DEBUG) << "freeing resources";
  CUDA_SAFE_CALL(cuModuleUnload(_module));
  CUDA_SAFE_CALL(cuCtxDestroy(_context));

  return *this;
}
