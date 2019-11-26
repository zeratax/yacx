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
  int minor() const { return _minor; }
  int major() const { return _major; }
  std::string name() const { return _name; }

 private:
  int _minor, _major;
  std::string _name;
  CUdevice _device;
};

} // namespace cudaexecutor

