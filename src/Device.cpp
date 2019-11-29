#include "cudaexecutor/Device.hpp"
#include "cudaexecutor/Exception.hpp"

using cudaexecutor::Device;

Device::Device() {
  CUDA_SAFE_CALL(cuInit(0));
  CUDA_SAFE_CALL(cuDeviceGet(&_device, 0));

  char cname[50];
  CUDA_SAFE_CALL(cuDeviceGetName(cname, 50, _device));
  _name = cname;

  CUDA_SAFE_CALL(cuDeviceGetAttribute(
      &_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, _device));
  CUDA_SAFE_CALL(cuDeviceGetAttribute(
      &_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, _device));
}
cudaexecutor::Device::Device(std::string name) {

}
