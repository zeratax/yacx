#pragma once

#include <string>
#include <vector>

#include "Headers.hpp"
#include "Program.hpp"
#include "util.hpp"

#include <cuda.h>
#include <nvrtc.h>

namespace cudaexecutor {

/*!
  \class Source Source.hpp
  \brief Class to wrap kernel strings
*/
class Source {
 public:
  //!
  //! \param kernel_string
  //! \param headers Headers needed to compile the kernel string
  explicit Source(std::string kernel_string, Headers headers = Headers());
  ~Source();
  //! create a Program
  //! \param function_name kernel name in kernel string
  //! \return a Program
  Program program(const std::string &function_name);

 private:
  std::string _kernel_string;
  Headers _headers;
  nvrtcProgram *_prog = nullptr;
};

} // namespace cudaexecutor
