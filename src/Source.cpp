#include "cudaexecutor/Source.hpp"
#include "cudaexecutor/Logger.hpp"
#include "cudaexecutor/Program.hpp"

#include <nvrtc.h>

#include <memory>
#include <utility>

using cudaexecutor::Source, cudaexecutor::ProgramArg, cudaexecutor::Program,
    cudaexecutor::loglevel;

Source::Source(std::string kernel_string, Headers headers)
    : _kernel_string{std::move(kernel_string)}, _headers{std::move(headers)} {
  logger(loglevel::DEBUG) << "created a Source with program string:\n'''\n"
                          << _kernel_string << "\n'''";
  logger(loglevel::DEBUG) << "Source uses " << _headers.size() << " Headers.";
}

Source::~Source() {
  // exception in destructor??
  logger(loglevel::DEBUG) << "destroying Source";
}

Program Source::program(const std::string &kernel_name) {
  logger(loglevel::DEBUG) << "creating a program for function: " << kernel_name;
  auto _prog = std::make_unique<nvrtcProgram>(); // custom deleter??
  NVRTC_SAFE_CALL(nvrtcCreateProgram(_prog.get(),            // progam
                                     _kernel_string.c_str(), // buffer
                                     kernel_name.c_str(),    // name
                                     _headers.size(),        // numHeaders
                                     _headers.content(),     // headers
                                     _headers.names()));     // includeNames
  return Program{kernel_name, std::move(_prog)};
}