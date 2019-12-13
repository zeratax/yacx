#include "yacx/Source.hpp"
#include "yacx/Logger.hpp"
#include "yacx/Program.hpp"

#include <nvrtc.h>

#include <memory>
#include <utility>

using yacx::Source, yacx::KernelArg, yacx::Program,
    yacx::loglevel;

Source::Source(std::string kernel_string, Headers headers)
    : m_kernel_string{std::move(kernel_string)}, m_headers{std::move(headers)} {
  logger(loglevel::DEBUG) << "created a Source with program string:\n'''\n"
                          << m_kernel_string << "\n'''";
  logger(loglevel::DEBUG) << "Source uses " << m_headers.numHeaders() << " Headers.";
}

Source::~Source() {
  // exception in destructor??
  logger(loglevel::DEBUG) << "destroying Source";
}

Program Source::program(const std::string &kernel_name) {
  logger(loglevel::DEBUG) << "creating a program for function: " << kernel_name;
  auto _prog = std::make_unique<nvrtcProgram>(); // custom deleter??
  NVRTC_SAFE_CALL(nvrtcCreateProgram(_prog.get(),            // progam
                                     m_kernel_string.c_str(), // buffer
                                     kernel_name.c_str(),    // name
                                     m_headers.numHeaders(),        // numHeaders
                                     m_headers.content(),     // headers
                                     m_headers.names()));     // includeNames
  return Program{kernel_name, std::move(_prog)};
}
