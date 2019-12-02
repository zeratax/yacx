#pragma once

#include "Device.hpp"
#include "KernelArg.hpp"
#include "Logger.hpp"

#include <cuda.h>
#include <memory>
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
  //! create a Kernel based on a templated kernel string
  //! \param _ptx
  //! \param kernel_name
  //! \param demangled_name
  Kernel(std::shared_ptr<char[]> _ptx, std::string demangled_name);
  //!
  //! \param grid vector of grid dimensions
  //! \param block vector of block dimensions
  //! \return this (for method chaining)
  Kernel &configure(dim3 grid, dim3 block);
  //!
  //! \param program_args
  //! \return this (for method chaining)
  Kernel &launch(std::vector<KernelArg> program_args, Device device = Device());

 private:
  std::shared_ptr<char[]> m_ptx; // shared pointer?
  std::string m_demangled_name;

  dim3 m_grid, m_block;
  CUcontext m_context;
  CUmodule m_module;
  CUfunction m_kernel;
};

} // namespace cudaexecutor
