#pragma once

#include "JNIHandle.hpp"

#include <cuda.h>
#include <vector_types.h>
#include <string>
#include <vector>
#include <functional>

namespace yacx {

class Device : JNIHandle {
  friend class Devices;
  /*!
    \class Device Device.hpp
    \brief Class to help get a CUDA-capable device
    for more info see: <a
    href="https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g9c3e1414f0ad901d3278a4d6645fc266">CUDA
    Driver API documentation</a>
  */
 public:
  //! Minor compute capability version number
  //! \return version number
  [[nodiscard]] int minor_version() const { return m_minor; }
  //! Major compute capability version number
  //! \return version number
  [[nodiscard]] int major_version() const { return m_major; }
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
  //! uuid for the device
  //! \return 16-byte UUID of the device as hexadecimal string
  std::string uuid() const { return m_uuid.bytes; } //TODO
  //!
  //! \param block returns block with maximum dimension
  void max_block_dim(dim3 *block);
  //!
  //! \param grid returns grid with maximum dimension
  void max_grid_dim(dim3 *grid);
  size_t max_shared_memory_per_block() const {
    return m_max_shared_memory_per_block;
  }

  //! Number of multiprocessors on device
  //! \return number of multiprocessors
  size_t multiprocessor_count() const { return m_multiprocessor_count; }
  //! Peak clock frequency in kilohertz
  //! \return peak clock frequency
  int clock_rate() const { return m_clock_rate; }
  //! Peak memory clock frequency in kilohertz
  //! \return peak memory clock frequency
  int memory_clock_rate() const { return m_memory_clock_rate; }
  //! Global memory bus width in bits
  //! \return bus width
  int bus_width() const { return m_bus_width; }
  //! Returns information about the device, see
  //! <a
  //! href=https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g9c3e1414f0ad901d3278a4d6645fc266>CUdevice_attribute</a>
  //! \param attrib CUDA device attribute
  //! \return
  int attribute(CUdevice_attribute attrib) const;

 private:
  //! Constructs a Device with the CUDA capable device with passed devicenumber
  Device(int ordinal);
  void set_device_properties(const CUdevice &device);
  std::string uuidToHex();

  int m_minor, m_major;
  std::string m_name;
  CUdevice m_device;
  CUuuid m_uuid;
  std::string m_uuidHex;
  size_t m_memory, m_max_shared_memory_per_block, m_multiprocessor_count;
  int m_clock_rate, m_memory_clock_rate, m_bus_width;
};

class Devices : JNIHandle {
  /*!
    \class Devices Devices.hpp
    \brief Class to help find CUDA-capable devices
    for more info see: <a
    href="https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g9c3e1414f0ad901d3278a4d6645fc266">CUDA
    Driver API documentation</a>
  */
  public:
    //! \return returns a Device with the first CUDA capable device it finds
    static Device& findDevice();

    //! \return returns a Device if a CUDA capable device with the identifier is
    //! available
    //! \param name Name of the cuda device, e.g.'Tesla K20c'
    static Device& findDevice(std::string name);

    //! \return returns a Device if a CUDA capable device with the passed 16-byte UUID as hexadecimal string
    //! \param uuid UUID of the cuda device
    static Device& findDeviceByUUID(std::string uuid);

    //! \return vector with all CUDA-capable devices
    static std::vector<Device*> findDevices();

    //! filters the devices satisfying passed condition
    //! \param con condition for devices e.g.'[](Device* d){return d->total_memory() >= 1024;}'
    //! \return list of devices satisfying passed condition
    static std::vector<Device*> findDevices(std::function<bool(Device*)> con);

  private:
    Devices();
    ~Devices();
    std::vector<Device*> m_devices;
    static Devices* getInstance();
    static Devices* instance;
};
}