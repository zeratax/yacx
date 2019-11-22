#pragma once

#include <cuda.h>
#include <string>

namespace cudaexecutor {
class Device {
  int _minor, _major;
  std::string _name;
  CUdevice _device;

 public:
  Device();
  int minor() const { return _minor; }
  int major() const { return _major; }
  std::string name() const { return _name; }
};

} // namespace cudaexecutor

#endif // CUDAEXECUTOR_DEVICE_HPP_
