#include "cudaexecutor/Device.hpp"
#include "cudaexecutor/Exception.hpp"

using cudaexecutor::Device, cudaexecutor::CUresultException;

Device::Device() {
  CUdevice device;
  CUDA_SAFE_CALL(cuInit(0));
  CUDA_SAFE_CALL(cuDeviceGet(&device, 0));
  this->set_device_properties(device);
}
cudaexecutor::Device::Device(std::string name) {
  int number{};
  char cname[50];
  CUdevice device;

  CUDA_SAFE_CALL(cuInit(0));
  CUDA_SAFE_CALL(cuDeviceGetCount(&number));

  for (int i{0}; i < number; ++i) {
    CUDA_SAFE_CALL(cuDeviceGet(&device, i));
    CUDA_SAFE_CALL(cuDeviceGetName(cname, 50, device));
    if(name == std::string{cname}) {
      this->set_device_properties(device);
      return;
    }
  }
  throw std::invalid_argument("Could not find device with this name!");
}


void Device::set_device_properties(const CUdevice &device) {
  _device = device;
  char cname[50];
  CUDA_SAFE_CALL(cuDeviceGetName(cname, 50, _device));
  _name = cname;

  CUDA_SAFE_CALL(cuDeviceGetAttribute(
      &_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, _device));
  CUDA_SAFE_CALL(cuDeviceGetAttribute(
      &_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, _device));
  CUDA_SAFE_CALL(cuDeviceTotalMem(&_memory, _device));
}