#pragma once

#include "ProgramArg.hpp"
#include <cudaexecutor/Logger.hpp>

#include <cuda.h>
#include <nvrtc.h>
#include <vector>
#include <vector_types.h>

namespace cudaexecutor {
/*!
  \class Kernel Kernel.hpp
  \brief Class to help launch and configure a CUDA kernel
  \example kernel_launch.cpp
*/
class Kernel {
 public:
  //TODO: kernel should only need PTX, nvrtcProgram and a kernel name
  explicit Kernel(char *_ptx, std::vector<std::string> template_parameters,
                  std::string kernel_name, std::string name_expression,
                  nvrtcProgram prog);
  //!
  //! \param grid vector of grid dimensions
  //! \param block vector of block dimensions
  //! \return this (for method chaining)
  Kernel &configure(dim3 grid, dim3 block);
  //!
  //! \param program_args
  //! \return this (for method chaining)
  Kernel &launch(std::vector<ProgramArg> program_args);

 private:
  char *_ptx; // shared pointer?
  std::vector<std::string> _template_parameters;
  std::string _kernel_name, _name_expression;
  nvrtcProgram _prog;

  dim3 _grid, _block;
  CUdevice _cuDevice;
  CUcontext _context;
  CUmodule _module;
  CUfunction _kernel;
};

} // namespace cudaexecutor
