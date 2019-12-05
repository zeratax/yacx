#pragma once

#include <cuda.h>
#include <string>
#include <vector_types.h>

#include "JNIHandle.hpp"

namespace cudaexecutor {

class Device : JNIHandle {
  /*!
    \class Device Device.hpp
    \brief Class to help get a CUDA-capable device
    for more info see: <a
    href="https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g9c3e1414f0ad901d3278a4d6645fc266">CUDA
    Driver API documentation</a>
  */
 public:
  //! Constructs a Device with the first CUDA capable device it finds
  Device();
  //! Constructs a Device if a CUDA capable device with the identifier is
  //! available
  //! \param name Name of the cuda device, e.g.'Tesla K20c'
  explicit Device(std::string name);
  //! Minor compute capability version number
  //! \return version number
  [[nodiscard]] int minor() const { return m_minor; }
  //! Major compute capability version number
  //! \return version number
  [[nodiscard]] int major() const { return m_major; }
  //! identifer string for the device
  //! \return identifer string
  std::string name() const { return m_name; }
  //!
  //! \return
  CUdevice get() const { return m_device; }
  //! Memory available on device for __constant__ variables in a CUDA C kernel
  //! in bytes
  //! \return Memory in bytes
  size_t total_memory() const { return m_memory; }
  //!
  //! \param block returns block with maximum dimension
  void max_block_dim(dim3 *block);
  //!
  //! \param grid returns grid with maximum dimension
  void max_grid_dim(dim3 *grid);
  size_t max_shared_memory_per_block() const {
    return m_max_shared_memory_per_block;
  }
  size_t multiprocessor_count() const { return m_multiprocessor_count; }
  int clock_rate() const { return m_clock_rate; }
  int memory_clock_rate() const { return m_memory_clock_rate; }
  int bus_width() const { return m_bus_width; }
  //! Returns information about the device, see
  //! <a
  //! href=https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g9c3e1414f0ad901d3278a4d6645fc266>CUdevice_attribute</a>
  //! \param attrib CUDA device attribute
  //! \return
  int attribute(CUdevice_attribute attrib) const;

 private:
  void set_device_properties(const CUdevice &device);

  int m_minor, m_major;
  std::string m_name;
  CUdevice m_device;
  size_t m_memory, m_max_shared_memory_per_block, m_multiprocessor_count;
  int m_clock_rate, m_memory_clock_rate, m_bus_width;
};

} // namespace cudaexecutor
