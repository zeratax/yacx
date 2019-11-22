#pragma once

#include <string>
#include <vector>

#include "Exception.hpp"
#include "Logger.hpp"
#include "Options.hpp"
#include "ProgramArg.hpp"
#include "Kernel.hpp"

#include <cuda.h>
#include <nvrtc.h>
#include <vector_types.h>

namespace cudaexecutor {

class Program {
  char *_ptx; // shared pointer?
  std::vector<std::string> _template_parameters;
  std::string _kernel_name, _name_expression, _log;
  nvrtcProgram _prog;

 public:
  Program(std::string function_name, nvrtcProgram prog);
  ~Program();
  template <typename T> Program &instantiate(T type);
  template <typename T, typename... TS> Program &instantiate(T type, TS... types);
  Kernel compile(const Options &options = Options());
  [[nodiscard]] std::string log() const { return _log; }
};

template <typename T> Program &Program::instantiate(T type) {
  _template_parameters.push_back(type);
  logger(cudaexecutor::loglevel::DEBUG1) << "adding last parameter " << type;
  return *this;
}

template <typename T, typename... TS>
Program &Program::instantiate(T type, TS... types) {
  _template_parameters.push_back(type);
  logger(cudaexecutor::loglevel::DEBUG1) << "adding parameter " << type;
  return Program::instantiate(types...);
}

} // namespace cudaexecutor

