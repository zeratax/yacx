#include "cudaexecutor/Device.hpp"
#include "cudaexecutor/Exception.hpp"
#include "cudaexecutor/Logger.hpp"

#include <experimental/iterator>
#include <vector>

using cudaexecutor::Device, cudaexecutor::CUresultException,
    cudaexecutor::loglevel;

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
  std::vector<std::string> devices;

  CUDA_SAFE_CALL(cuInit(0));
  CUDA_SAFE_CALL(cuDeviceGetCount(&number));

  for (int i{0}; i < number; ++i) {
    CUDA_SAFE_CALL(cuDeviceGet(&device, i));
    CUDA_SAFE_CALL(cuDeviceGetName(cname, 50, device));
    if (name == std::string{cname}) {
      this->set_device_properties(device);
      return;
    }
    devices.push_back(std::string{cname});
  }
  std::ostringstream buffer;
  std::copy(devices.begin(), devices.end(),
            std::experimental::make_ostream_joiner(buffer, ", "));
  throw std::invalid_argument(
      "Could not find device with this name! Available devices : " +
      buffer.str());
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

void Device::max_block_dim(dim3 *block) {
  int x, y, z;
  CUDA_SAFE_CALL(
      cuDeviceGetAttribute(&x, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, _device));
  block->x = x;
  logger(loglevel::DEBUG1) << "block.x = " << block->x;
  CUDA_SAFE_CALL(
      cuDeviceGetAttribute(&y, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, _device));
  block->y = y;
  logger(loglevel::DEBUG1) << "block.y = " << block->y;
  CUDA_SAFE_CALL(
      cuDeviceGetAttribute(&z, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, _device));
  block->z = z;
  logger(loglevel::DEBUG1) << "block.z = " << block->z;
}

void Device::max_grid_dim(dim3 *grid) {
  int x, y, z;
  CUDA_SAFE_CALL(
      cuDeviceGetAttribute(&x, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, _device));
  grid->x = x;
  logger(loglevel::DEBUG1) << "grid.x = " << grid->x;
  CUDA_SAFE_CALL(
      cuDeviceGetAttribute(&y, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, _device));
  grid->y = y;
  logger(loglevel::DEBUG1) << "grid.y = " << grid->y;
  CUDA_SAFE_CALL(
      cuDeviceGetAttribute(&z, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, _device));
  grid->z = z;
  logger(loglevel::DEBUG1) << "grid.z = " << grid->z;
}