#include "../include/cudaexecutor/Device.hpp"
#include "../include/cudaexecutor/Exception.hpp"

#define CUDA_SAFE_CALL(x)                                                      \
  do {                                                                         \
    CUresult result = x;                                                       \
    if (result != CUDA_SUCCESS) {                                              \
      throw cuda_exception(result, std::string(__FILE__), __LINE__);           \
    }                                                                          \
  } while (0)

using cudaexecutor::Device;

Device::Device() {
  CUDA_SAFE_CALL(cuInit(0));
  CUDA_SAFE_CALL(cuDeviceGet(&_device, 0));

  char *cname = new char[50]; // destruktor??
  CUDA_SAFE_CALL(cuDeviceGetName(cname, 50, _device));
  _name = cname;
  std::free(cname);

  CUDA_SAFE_CALL(cuDeviceGetAttribute(
      &_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, _device));
  CUDA_SAFE_CALL(cuDeviceGetAttribute(
      &_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, _device));
}
