#include "cudaexecutor/Program.hpp"
#include "cudaexecutor/Exception.hpp"
#include "cudaexecutor/Headers.hpp"
#include "cudaexecutor/Logger.hpp"
#include "cudaexecutor/util.hpp"

#include <experimental/iterator>
#include <iostream>
#include <memory>
#include <utility>

using cudaexecutor::Program, cudaexecutor::Kernel, cudaexecutor::Options,
    cudaexecutor::Headers, cudaexecutor::ProgramArg, cudaexecutor::loglevel,
    cudaexecutor::detail::whichError, cudaexecutor::detail::descriptionFkt;

Program::Program(std::string kernel_name, nvrtcProgram prog)
    : _kernel_name{std::move(kernel_name)}, _prog{prog} {
  logger(loglevel::DEBUG) << "created Program " << _kernel_name;
  CUDA_SAFE_CALL(cuInit(0));
}

Program::~Program() {
  // Exceptions in destructor usually a bad idea??
  // Release resources.
  logger(loglevel::DEBUG) << "destroying Program " << _kernel_name;
  nvrtcResult error = nvrtcDestroyProgram(&_prog);
  if (error != NVRTC_SUCCESS) {
    auto description = whichError(error);
    std::cout << descriptionFkt(description) << std::endl;
  }
}

Kernel Program::compile(const Options &options) {
  logger(loglevel::INFO) << "compiling Program";
  if (!_template_parameters.empty()) {
    logger(loglevel::DEBUG) << "with following template parameters:";
    for (auto &parameter : _template_parameters)
      logger(loglevel::DEBUG) << parameter;

    std::ostringstream buffer;
    std::copy(_template_parameters.begin(), _template_parameters.end(),
              std::experimental::make_ostream_joiner(buffer, ", "));
    _name_expression = _kernel_name + "<" + buffer.str() + ">";

    logger(loglevel::DEBUG)
        << "which results in the following name expression: "
        << _name_expression;
    NVRTC_SAFE_CALL(nvrtcAddNameExpression(_prog, _name_expression.c_str()));
  } else {
    logger(loglevel::DEBUG1) << "with no template parameters";
  }

  nvrtcResult compileResult =
      nvrtcCompileProgram(_prog, options.numOptions(), options.options());

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
  _ptx = new char[ptxSize]; // shared pointer??
  NVRTC_SAFE_CALL(nvrtcGetPTX(_prog, _ptx));

  logger(loglevel::INFO) << "Program compiled";
  // lowered name
  const char *name = _kernel_name.c_str(); // copy??
  if (!_name_expression.empty()) {
    logger(loglevel::DEBUG) << "getting lowered name for function";
    NVRTC_SAFE_CALL(nvrtcGetLoweredName(_prog, _name_expression.c_str(), &name))
  }
  // templated kernel string needs to be demangled to launch
  return Kernel{_ptx, name};
}
