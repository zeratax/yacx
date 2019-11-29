#pragma once

#include "ProgramArg.hpp"
#include "Logger.hpp"

#include <cuda.h>
#include <nvrtc.h>
#include <vector>
#include <vector_types.h>
#include <memory>

namespace cudaexecutor {
/*!
  \class Kernel Kernel.hpp
  \brief Class to help launch and configure a CUDA kernel
  \example kernel_launch.cpp
*/
class Kernel {
 public:
  //! create a Kernel based on a templated kernel string
  //! \param _ptx
  //! \param kernel_name
  //! \param demangled_name
  Kernel(std::shared_ptr<char[]> _ptx, const char* demangled_name);
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
  std::shared_ptr<char[]> _ptx; // shared pointer?
  const char* _demangled_name;

  dim3 _grid, _block;
  CUdevice _cuDevice;
  CUcontext _context;
  CUmodule _module;
  CUfunction _kernel;
};

} // namespace cudaexecutor
