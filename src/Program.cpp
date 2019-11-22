#include "cudaexecutor/Program.hpp"
#include "../include/cudaexecutor/Exception.hpp"
#include "../include/cudaexecutor/Headers.hpp"
#include "../include/cudaexecutor/Logger.hpp"
#include "../include/cudaexecutor/util.hpp"

#include <memory>
#include <utility>

using cudaexecutor::Program, cudaexecutor::Kernel, cudaexecutor::Options, cudaexecutor::Headers,
    cudaexecutor::ProgramArg, cudaexecutor::to_comma_separated,
    cudaexecutor::loglevel;

Program::Program(std::string kernel_name, nvrtcProgram prog)
    : _kernel_name{std::move(kernel_name)}, _prog{prog} {
  logger(loglevel::DEBUG) << "created Program " << _kernel_name;
  CUDA_SAFE_CALL(cuInit(0));
}

Program::~Program() {
  // Exceptions in destructor usually a bad idea??
  // Release resources.
  logger(loglevel::DEBUG) << "destroying Program " << _kernel_name;
  //  CUDA_SAFE_CALL(cuModuleUnload(_module));
  //  CUDA_SAFE_CALL(cuCtxDestroy(_context));
}

Kernel Program::compile(const Options &options) {
  logger(loglevel::INFO) << "compiling Program";
  if (!_template_parameters.empty()) {
    logger(loglevel::DEBUG) << "with following template parameters:";
    for (auto &parameter : _template_parameters)
      logger(loglevel::DEBUG) << parameter;

    _name_expression =
        _kernel_name + "<" + to_comma_separated(_template_parameters) + ">";
    logger(loglevel::DEBUG)
        << "which results in the following name expression: "
        << _name_expression;
    NVRTC_SAFE_CALL(nvrtcAddNameExpression(_prog, _name_expression.c_str()));
  } else {
    logger(loglevel::DEBUG1) << "with no template parameters";
  }

  nvrtcResult compileResult =
      nvrtcCompileProgram(_prog, options.numOptions(), options.options());

  logger(loglevel::DEBUG) << "Program compiled";

  size_t logSize;
  NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(_prog, &logSize));
  auto clog = std::make_unique<char[]>(logSize);
  NVRTC_SAFE_CALL(nvrtcGetProgramLog(_prog, clog.get()));
  _log = clog.get();

  if (compileResult != NVRTC_SUCCESS) {
    logger(loglevel::ERROR) << _log;
    NVRTC_SAFE_CALL(compileResult);
  }


  size_t ptxSize;
  NVRTC_SAFE_CALL(nvrtcGetPTXSize(_prog, &ptxSize));
  _ptx = new char[ptxSize];
  NVRTC_SAFE_CALL(nvrtcGetPTX(_prog, _ptx));

  logger(loglevel::INFO) << "Program compiled";
  return Kernel {_ptx, _template_parameters, _kernel_name, _name_expression, _prog};
}
