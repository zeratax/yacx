#include "../include/cudaexecutor/Program.hpp"
#include "../include/cudaexecutor/Logger.hpp"
#include "../include/cudaexecutor/Kernel.hpp"

#include <nvrtc.h>

#include <utility>

using cudaexecutor::Program, cudaexecutor::ProgramArg, cudaexecutor::Kernel,
cudaexecutor::loglevel;

Program::Program(std::string kernel_string, Headers headers)
        : _kernel_string{std::move(kernel_string)}, _headers{std::move(headers)} {
    logger(loglevel::DEBUG) << "created a Program with kernel string:\n'''\n"
                            << _kernel_string << "\n'''";
    logger(loglevel::DEBUG) << "Program uses " << headers.size() << " Headers.";
}

Program::~Program() {
    // exception in destructor??
    logger(loglevel::DEBUG) << "destroying Program";
    // NVRTC_SAFE_CALL(nvrtcDestroyProgram(_prog));
    nvrtcDestroyProgram(_prog);
}

Kernel Program::kernel(const std::string &function_name) {
    logger(loglevel::DEBUG) << "creating a kernel for function: "
                            << function_name;
    _prog = new nvrtcProgram;                  // destructor?
    nvrtcCreateProgram(_prog,                  // prog
                       _kernel_string.c_str(), // buffer
                       function_name.c_str(),  // name
                       _headers.size(),        // numHeaders
                       _headers.content(),     // headers
                       _headers.names());      // includeNames
    return Kernel(function_name, *_prog);
}
