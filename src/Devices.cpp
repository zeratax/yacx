#include "yacx/Devices.hpp"

#include "yacx/Exception.hpp"
#include "yacx/Init.hpp"

#include <algorithm>
#include <iostream>
#include <sstream>

using yacx::Device, yacx::Devices;

Devices *Devices::instance = NULL;

Devices *Devices::getInstance() {
  if (instance == NULL) {
    instance = new Devices();
  }

  return instance;
}

Devices::Devices() {
  int number{};

  yacx::detail::init();
  CUDA_SAFE_CALL(cuDeviceGetCount(&number));

  m_devices.reserve(number);

  if (number == 0) {
    throw std::invalid_argument("no CUDA capable device found!");
  }

  for (int i{0}; i < number; ++i) {
    m_devices.emplace_back(Device(i));
  }
}

Device &Devices::findDevice() { 
  return getInstance()->m_devices[0]; }

Device &Devices::findDevice(std::string name) {
  std::vector<Device *> devices =
      findDevices([name](Device& device) { return device.m_name == name; });

  if (!devices.empty()) {
    return *(devices[0]);
  } else {
    std::vector<Device> devices = getInstance()->m_devices;
    std::stringstream buffer;
    buffer << "Could not find device with this name! Available devices: [";
    for (unsigned int i{0}; i < devices.size() - 1; ++i) {
      buffer << devices[i].m_name << ", ";
    }
    buffer << devices[devices.size() - 1].m_name;
    buffer << "]";

    throw std::invalid_argument(buffer.str());
  }
}

Device &Devices::findDeviceByUUID(std::string uuid) {
  // Delete all '-' in uuid
  uuid.erase(std::remove(uuid.begin(), uuid.end(), '-'), uuid.end());

  std::vector<Device *> devices =
      findDevices([uuid](Device& device) { return device.uuid() == uuid; });

  if (!devices.empty()) {
    return *(devices[0]);
  } else {
    std::vector<Device>& devices = getInstance()->m_devices;
    std::stringstream buffer;
    buffer
        << "Could not find device with this name! Available UUIDs-devices: [";
    for (unsigned int i{0}; i < devices.size() - 1; ++i) {
      buffer << devices[i].uuid() << ", ";
    }
    buffer << devices[devices.size() - 1].uuid();
    buffer << "]";

    throw std::invalid_argument(buffer.str());
  }
}

std::vector<Device>& Devices::findDevices() {
  return getInstance()->m_devices;
}

std::vector<Device *> Devices::findDevices(std::function<bool(Device&)> con) {
  std::vector<Device>& devices = getInstance()->m_devices;

  std::vector<Device *> filterd;
  filterd.reserve(devices.size());

  for (unsigned int i = 0; i < devices.size(); ++i){
    if (con(devices[i])){
      filterd.emplace_back(&devices[i]);
    }
  }

  return filterd;
}