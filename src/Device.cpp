#include "yacx/Devices.hpp"
#include "yacx/Exception.hpp"
#include "yacx/Logger.hpp"

#include <iostream>
#include <sstream>

using yacx::Device, yacx::CUresultException, yacx::loglevel;

Device::Device(int ordinal) {
  CUdevice device;
  CUDA_SAFE_CALL(cuDeviceGet(&device, ordinal));
  this->set_device_properties(device);
}

void Device::set_device_properties(const CUdevice &device) {
  m_device = device;
  char cname[50];
  CUDA_SAFE_CALL(cuDeviceGetName(cname, 50, m_device));
  m_name = cname;

  m_major = attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
  m_minor = attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
  m_max_shared_memory_per_block =
      attribute(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK);
  m_multiprocessor_count = attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
  m_clock_rate = attribute(CU_DEVICE_ATTRIBUTE_CLOCK_RATE);
  m_memory_clock_rate = attribute(CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE);
  m_bus_width = attribute(CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH);

  CUDA_SAFE_CALL(cuDeviceTotalMem(&m_memory, m_device));

  CUDA_SAFE_CALL(cuDeviceGetUuid(&m_uuid, m_device));
  m_uuidHex = uuidToHex();
}

std::string Device::uuidToHex(){
  std::stringstream ss;
  for(int i = 0; i < 16; i++) {
    ss << std::hex << (int)m_uuid.bytes[i];
  }
  return ss.str();
}

void Device::max_block_dim(dim3 *block) {
  block->x = attribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X);
  logger(loglevel::DEBUG1) << "block.x = " << block->x;

  block->y = attribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y);
  logger(loglevel::DEBUG1) << "block.y = " << block->y;

  block->z = attribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z);
  logger(loglevel::DEBUG1) << "block.z = " << block->z;
}

void Device::max_grid_dim(dim3 *grid) {
  grid->x = attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X);
  logger(loglevel::DEBUG1) << "grid.x = " << grid->x;

  grid->y = attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y);
  logger(loglevel::DEBUG1) << "grid.y = " << grid->y;

  grid->z = attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z);
  logger(loglevel::DEBUG1) << "grid.z = " << grid->z;
}

int Device::attribute(CUdevice_attribute attrib) const {
  int pi{0};
  CUDA_SAFE_CALL(cuDeviceGetAttribute(&pi, attrib, m_device));
  return pi;
}