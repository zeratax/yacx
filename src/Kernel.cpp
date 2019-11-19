#include "../include/cudaexecutor/Kernel.hpp"
#include "../include/cudaexecutor/Exception.hpp"
#include "../include/cudaexecutor/Headers.hpp"
#include "../include/cudaexecutor/Logger.hpp"
#include "../include/cudaexecutor/util.hpp"

#include <memory>
#include <utility>

using cudaexecutor::Kernel, cudaexecutor::Options, cudaexecutor::Headers,
    cudaexecutor::ProgramArg, cudaexecutor::to_comma_separated,
    cudaexecutor::loglevel;

Kernel::Kernel(std::string kernel_name, nvrtcProgram *prog)
    : _kernel_name{std::move(kernel_name)}, _prog{prog} {
  logger(loglevel::DEBUG) << "created Kernel " << _kernel_name;
  CUDA_SAFE_CALL(cuInit(0));
}

Kernel::~Kernel() {
  // Exceptions in destructor usually a bad idea??
  // Release resources.
  logger(loglevel::DEBUG) << "destroying Kernel " << _kernel_name;
  //  CUDA_SAFE_CALL(cuModuleUnload(_module));
  //  CUDA_SAFE_CALL(cuCtxDestroy(_context));
}

Kernel Kernel::configure(dim3 grid, dim3 block) {
  logger(loglevel::DEBUG) << "configuring Kernel with grid: " << grid.x << ", "
                          << grid.y << ", " << grid.z << " and block "
                          << block.x << ", " << block.y << ", " << block.z;
  _grid = grid;
  _block = block;
  return *this;
}

Kernel Kernel::launch(std::vector<ProgramArg> args) {
  logger(loglevel::DEBUG) << "launching Kernel";
  if (!_compiled)
    throw std::invalid_argument("kernel needs to be compiled to be launched");

  // check if device already initialised
  CUDA_SAFE_CALL(cuDeviceGet(&_cuDevice, 0));

  CUDA_SAFE_CALL(cuCtxCreate(&_context, 0, _cuDevice));
  CUDA_SAFE_CALL(cuModuleLoadDataEx(&_module, _ptx, 0, nullptr, nullptr));

  logger(loglevel::DEBUG) << "uploading arguemnts";
  void *kernel_args[args.size()];
  int i{0};
  for (auto &arg : args) {
    arg.upload();
    kernel_args[i++] = arg.content();
  }

  // lowered name
  const char *name = _kernel_name.c_str();
  if (!_template_parameters.empty()) {
    logger(loglevel::DEBUG) << "getting lowered name for function";
    NVRTC_SAFE_CALL(
        nvrtcGetLoweredName(*_prog, _name_expression.c_str(), &name));
  }
  CUDA_SAFE_CALL(cuModuleGetFunction(&_kernel, _module, name));

  // launch the kernel

  logger(loglevel::INFO) << "launching " << name << "<" << _name_expression << ">";
  CUDA_SAFE_CALL(cuLaunchKernel(_kernel, // function from kernel
                                _grid.x, _grid.y, _grid.z,    // grid dim
                                _block.x, _block.y, _block.z, // block dim
                                0, nullptr,             // shared mem and stream
                                kernel_args, nullptr)); // arguments
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

Kernel Kernel::compile(const Options &options) {
  logger(loglevel::INFO) << "compiling Kernel";
  if (!_template_parameters.empty()) {
    logger(loglevel::DEBUG)
        << "with following template parameters:";
    for (auto &parameter : _template_parameters)
      logger(loglevel::DEBUG) << parameter;

    _name_expression =
        _kernel_name + "<" + to_comma_separated(_template_parameters) + ">";
    logger(loglevel::DEBUG)
        << "which results in the following name expression: "
        << _name_expression;
    NVRTC_SAFE_CALL(nvrtcAddNameExpression(*_prog, _name_expression.c_str()));
  }

  nvrtcResult compileResult =
      nvrtcCompileProgram(*_prog, options.numOptions(), options.options());

  size_t logSize;
  NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(*_prog, &logSize));
  auto clog = std::make_unique<char[]>(logSize);
  NVRTC_SAFE_CALL(nvrtcGetProgramLog(*_prog, clog.get()));
  _log = clog.get();

  if (compileResult != NVRTC_SUCCESS)
    logger(loglevel::ERROR) << _log;
  NVRTC_SAFE_CALL(compileResult);

  size_t ptxSize;
  NVRTC_SAFE_CALL(nvrtcGetPTXSize(*_prog, &ptxSize));
  _ptx = new char[ptxSize];
  NVRTC_SAFE_CALL(nvrtcGetPTX(*_prog, _ptx));

  logger(loglevel::INFO) << "Kernel compiled";
  _compiled = true;
  return *this;
}
