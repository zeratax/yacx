#pragma once

#include <cuda.h>
#include <string>

namespace cudaexecutor {

class Device {
  /*!
    \class Device Device.hpp
    \brief Class to help get a CUDA-capable device
  */
 public:
  Device();
  explicit Device(std::string name);
  [[nodiscard]] int minor() const { return _minor; }
  [[nodiscard]] int major() const { return _major; }
  std::string name() const { return _name; }
  CUdevice device() { return _device; }

 private:
  int _minor, _major;
  std::string _name;
  CUdevice _device;
};

} // namespace cudaexecutor
