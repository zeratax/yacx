#pragma once

#include <string>
#include <vector>

#include "Exception.hpp"
#include "Kernel.hpp"
#include "Logger.hpp"
#include "Options.hpp"
#include "ProgramArg.hpp"

#include <cuda.h>
#include <nvrtc.h>
#include <vector_types.h>

namespace cudaexecutor {

/*!
  \class Program Program.hpp
  \brief Class to instantiate and compile Source (kernel strings)
*/
class Program {
 public:
  //!
  //! \param function_name function name in kernel string
  //! \param prog
  Program(std::string function_name, nvrtcProgram prog);
  ~Program();
  //! instantiate template parameter
  //! \tparam T
  //! \param type
  //! \return
  template <typename T> Program &instantiate(T type);
  //! instantiate template parameters
  //! \tparam T
  //! \tparam TS
  //! \param type
  //! \param types
  //! \return
  template <typename T, typename... TS>
  Program &instantiate(T type, TS... types);
  //! compile Program to Kernel
  //! \param options see <a href="https://docs.nvidia.com/cuda/nvrtc/index.html#group__options">NVRTC documentation</a> for supported Options
  //! \return a compiled Kernel
  Kernel compile(const Options &options = Options());
  //!
  //! \return log of compilation
  [[nodiscard]] std::string log() const { return _log; }

 private:
  char *_ptx; // shared pointer?
  std::vector<std::string> _template_parameters;
  std::string _kernel_name, _name_expression, _log;
  nvrtcProgram _prog;
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
