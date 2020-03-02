#include "yacx/Devices.hpp"

#include "yacx/Exception.hpp"
#include "yacx/Init.hpp"

#include <algorithm>
#include <iostream>
#include <sstream>

using yacx::Device, yacx::Devices;

Devices* Devices::instance = NULL;

Devices* Devices::getInstance(){
    if (instance == NULL){
        instance = new Devices();
    }

    return instance;
}

Devices::Devices(){
    int number{};

    yacx::detail::init();
    CUDA_SAFE_CALL(cuDeviceGetCount(&number));

    if (number == 0){
        throw std::invalid_argument("no CUDA capable device found!");
    }

    std::vector<Device*> devices;
    devices.resize(number);

    for (int i{0}; i < number; ++i) {
        m_devices.push_back(new Device(i));
    }
}

Devices::~Devices(){
    for (int i{0}; i < m_devices.size(); ++i) {
        delete m_devices[i];
    }
}

Device& Devices::findDevice(){
    return *(getInstance()->m_devices[0]);
}

Device& Devices::findDevice(std::string name){
    std::vector<Device*> devices = findDevices([name](Device* device){return device->m_name == name;});

    if (!devices.empty()){
        return *(devices[0]);
    } else {
        std::vector<Device*> devices = getInstance()->m_devices;
        std::stringstream buffer;
        buffer << "Could not find device with this name! Available devices: [";
        for (int i{0}; i < devices.size()-1; ++i) {
            buffer << devices[i]->m_name << ", ";
        }
        buffer << devices[devices.size()-1]->m_name;
        buffer << "]";

        throw std::invalid_argument(buffer.str());
    }
}

Device& Devices::findDeviceByUUID(std::string uuid){
    //Delete all '-' in uuid
    uuid.erase(std::remove(uuid.begin(), uuid.end(), '-'), uuid.end());

    std::vector<Device*> devices = findDevices([uuid](Device* device){return device->uuid() == uuid;});

    if (!devices.empty()){
        return *(devices[0]);
    } else {
        std::vector<Device*> devices = getInstance()->m_devices;
        std::stringstream buffer;
        buffer << "Could not find device with this name! Available UUIDs-devices: [";
        for (int i{0}; i < devices.size()-1; ++i) {
            buffer << devices[i]->uuid() << ", ";
        }
        buffer << devices[devices.size()-1]->uuid();
        buffer << "]";

        throw std::invalid_argument(buffer.str());
    }
}

std::vector<Device*> Devices::findDevices(){
    return getInstance()->m_devices; 
}

std::vector<Device*> Devices::findDevices(std::function<bool(Device*)> con){
    std::vector<Device*> filterd;
    std::copy_if(getInstance()->m_devices.begin(), getInstance()->m_devices.end(),
        std::back_inserter(filterd), con);
    return filterd;
}