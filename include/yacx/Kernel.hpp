#pragma once

#include "Device.hpp"
#include "JNIHandle.hpp"
#include "KernelArgs.hpp"
#include "KernelTime.hpp"
#include "Logger.hpp"

#include <cuda.h>
#include <memory>
#include <nvrtc.h>
#include <vector>
#include <vector_types.h>

namespace yacx {
/*!
  \class Kernel Kernel.hpp
  \brief Class to help launch and configure a CUDA kernel
  \example docs/kernel_launch.cpp
  \example example_saxpy.cpp
*/
class Kernel : JNIHandle {
 public:
  //! create a Kernel based on a templated kernel string
  //! \param ptx
  //! \param kernel_name
  //! \param demangled_name
  Kernel(std::shared_ptr<char[]> ptx, std::string demangled_name);
  //!
  //! \param grid vector of grid dimensions
  //! \param block vector of block dimensions
  //! \param shared amount of dynamic shared memory to allocate
  //! \return this (for method chaining)
  Kernel &configure(dim3 grid, dim3 block, unsigned int shared = 0);
  //!
  //! \param kernel_args
  //! \return KernelTime
  KernelTime launch(KernelArgs args, Device device = Device());
  //! benchmark a Kernel
  //! \param kernel_args
  //! \param number of executions
  //! \param device
  //! \return vector of KernelTimes for every execution
  std::vector<KernelTime> benchmark(KernelArgs args, unsigned int executions,
                                    Device device = Device());

 private:
  KernelTime launch(KernelArgs args, void *downloadDest);

  std::shared_ptr<char[]> m_ptx;
  std::string m_demangled_name;

  dim3 m_grid, m_block;
  unsigned int m_shared;
  CUcontext m_context;
  CUmodule m_module;
  CUfunction m_kernel;
};

} // namespace yacx
