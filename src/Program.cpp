#include "yacx/Program.hpp"
#include "yacx/Exception.hpp"
#include "yacx/Init.hpp"
#include "yacx/Logger.hpp"
#include "yacx/util.hpp"

#include <experimental/iterator>
#include <iostream>
#include <memory>
#include <utility>

using yacx::Program, yacx::Kernel, yacx::Options, yacx::KernelArg,
    yacx::loglevel, yacx::detail::whichError, yacx::detail::descriptionFkt;

Program::Program(std::string kernel_name, std::shared_ptr<nvrtcProgram> prog)
    : m_kernel_name{std::move(kernel_name)}, m_prog{std::move(prog)} {
  logger(loglevel::DEBUG) << "created Program " << m_kernel_name;
  yacx::detail::init();
}

Program::~Program() {
  // Exceptions in destructor usually a bad idea??
  // Release resources.
  logger(loglevel::DEBUG) << "destroying Program " << m_kernel_name;
  nvrtcResult error = nvrtcDestroyProgram(m_prog.get());
  if (error != NVRTC_SUCCESS) {
    auto description = whichError(error);
    logger(loglevel::ERROR)
        << "could not destroy program: " << descriptionFkt(description);
  }
}

Kernel Program::compile(const Options &options) {
  logger(loglevel::INFO) << "compiling Program";
  if (!m_template_parameters.empty()) {
    logger(loglevel::DEBUG) << "with following template parameters:";
    for (auto &parameter : m_template_parameters)
      logger(loglevel::DEBUG) << parameter;

    std::ostringstream buffer;
    std::copy(m_template_parameters.begin(), m_template_parameters.end(),
              std::experimental::make_ostream_joiner(buffer, ", "));
    m_name_expression = m_kernel_name + "<" + buffer.str() + ">";

    logger(loglevel::DEBUG)
        << "which results in the following name expression: "
        << m_name_expression;
    NVRTC_SAFE_CALL(nvrtcAddNameExpression(*m_prog, m_name_expression.c_str()));
  } else {
    logger(loglevel::DEBUG1) << "with no template parameters";
  }

  nvrtcResult compileResult =
      nvrtcCompileProgram(*m_prog, options.numOptions(), options.content());

  size_t logSize;
  NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(*m_prog, &logSize));
  auto clog = std::make_unique<char[]>(logSize);
  NVRTC_SAFE_CALL(nvrtcGetProgramLog(*m_prog, clog.get()));
  m_log = clog.get();

  if (compileResult != NVRTC_SUCCESS) {
    // logger(loglevel::ERROR) << m_log;
    NVRTC_SAFE_CALL_LOG(compileResult, m_log);
  }

  size_t ptxSize;
  NVRTC_SAFE_CALL(nvrtcGetPTXSize(*m_prog, &ptxSize));
  auto ptx = std::make_unique<char[]>(ptxSize);
  NVRTC_SAFE_CALL(nvrtcGetPTX(*m_prog, ptx.get()));

  logger(loglevel::INFO) << "Program compiled";
  // lowered name
  const char *name = m_kernel_name.c_str(); // copy??
  if (!m_name_expression.empty()) {
    logger(loglevel::DEBUG) << "getting lowered name for function";
    NVRTC_SAFE_CALL(
        nvrtcGetLoweredName(*m_prog, m_name_expression.c_str(), &name))
  }
  // templated kernel string needs to be demangled to launch
  return Kernel{std::move(ptx), std::string{name}};
}
