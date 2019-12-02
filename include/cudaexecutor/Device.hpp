#pragma once

#include <cuda.h>
#include <string>
#include <vector_types.h>

namespace cudaexecutor {

class Device {
  /*!
    \class Device Device.hpp
    \brief Class to help get a CUDA-capable device
  */
 public:
  Device();
  explicit Device(std::string name);
  [[nodiscard]] int minor() const { return m_minor; }
  [[nodiscard]] int major() const { return m_major; }
  std::string name() const { return m_name; }
  CUdevice get() { return m_device; }
  size_t total_memory() { return m_memory; }
  void max_block_dim(dim3 *block);
  void max_grid_dim(dim3 *grid);

 private:
  void set_device_properties(const CUdevice &device);

  int m_minor, m_major;
  std::string m_name;
  CUdevice m_device;
  size_t m_memory;
};

} // namespace cudaexecutor
