#pragma once

#include "Devices.hpp"
#include "JNIHandle.hpp"
#include "KernelArgs.hpp"
#include "KernelTime.hpp"
#include "Logger.hpp"

#include <builtin_types.h>
#include <cuda.h>
#include <functional>
#include <memory>
#include <nvrtc.h>
#include <vector>
#include <vector_types.h>

namespace yacx {
typedef struct {
  CUevent start;
  CUevent end;

  float elapsed();
} eventInterval;

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
  KernelTime launch(KernelArgs args, Device &device = Devices::findDevice());
  //! benchmark a Kernel
  //! \param kernel_args
  //! \param number of executions
  //! \param device
  //! \return vector of KernelTimes for every execution
  std::vector<KernelTime> benchmark(std::vector<KernelArg> &args,
                                    unsigned int executions,
                                    Device &device = Devices::findDevice());

 private:
  eventInterval asyncOperation(
      KernelArgs &args, CUstream stream, CUevent syncEvent,
      std::function<void(KernelArgs &args, CUstream stream)> operation);
  eventInterval uploadAsync(KernelArgs &args, Device &device);
  eventInterval runAsync(KernelArgs &args, Device &device, CUevent syncEvent);
  eventInterval downloadAsync(KernelArgs &args, Device &device,
                              CUevent syncEvent, void *downloadDest);

  struct KernelFunction {
    CUmodule module;
    CUfunction kernel;

    KernelFunction(char *ptx, std::string demangled_name);

    ~KernelFunction();
  };

  std::shared_ptr<char[]> m_ptx;
  std::shared_ptr<struct KernelFunction> m_kernelFunction;
  std::string m_demangled_name;

  dim3 m_grid, m_block;
  unsigned int m_shared;
};

} // namespace yacx
