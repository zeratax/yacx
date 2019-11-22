#include "cudaexecutor/Source.hpp"
#include "../include/cudaexecutor/Logger.hpp"
#include "cudaexecutor/Program.hpp"

#include <nvrtc.h>

#include <utility>

using cudaexecutor::Source, cudaexecutor::ProgramArg, cudaexecutor::Program,
    cudaexecutor::loglevel;

Source::Source(std::string kernel_string, Headers headers)
    : _kernel_string{std::move(kernel_string)}, _headers{std::move(headers)} {
  logger(loglevel::DEBUG) << "created a Source with program string:\n'''\n"
                          << _kernel_string << "\n'''";
  logger(loglevel::DEBUG) << "Source uses " << headers.size() << " Headers.";
}

Source::~Source() {
  // exception in destructor??
  logger(loglevel::DEBUG) << "destroying Source";
  NVRTC_SAFE_CALL(nvrtcDestroyProgram(_prog));
}

Program Source::program(const std::string &function_name) {
  logger(loglevel::DEBUG) << "creating a program for function: "
                          << function_name;
  _prog = new nvrtcProgram;                  // destructor?
  nvrtcCreateProgram(_prog,                  // prog
                     _kernel_string.c_str(), // buffer
                     function_name.c_str(),  // name
                     _headers.size(),        // numHeaders
                     _headers.content(),     // headers
                     _headers.names());      // includeNames
  return Program(function_name, *_prog);
}
