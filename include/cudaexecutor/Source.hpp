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
  Program program(const std::string &kernel_name);

 private:
  std::string m_kernel_string;
  Headers m_headers;
};

} // namespace cudaexecutor
