#include "yacx/Devices.hpp"

#include "yacx/Exception.hpp"
#include "yacx/Init.hpp"

#include <experimental/iterator>
#include <vector>

using yacx::Device, yacx::Devices;

Devices* Devices::instance = NULL;

Devices* Devices::getDevices(){
    if (instance == NULL){
        instance = new Devices();
    }

    return instance;
}

Devices Devices::Devices(){
    int number{};

    yacx::detail::init();
    CUDA_SAFE_CALL(cuDeviceGetCount(&number));

    std::vector<Device> devices;
    devices.resize(number);

    for (int i{0}; i < number; ++i) {
        m_devices.push_back(Device(i));
    }
}

Device Devices::findDevice(){
    return getDevices()->m_devices[0];
}

Device Devices::findDevice(std::string name){
    std::vector<Device> devices = getDevices()->filter([](Device device){return device.name == name;});

    if (!devices.empty()){
        return devices[0];
    } else {
    //TODO
    //     std::ostringstream buffer;
    //     std::copy(devices.begin(), devices.end(),
    //         std::experimental::make_ostream_joiner(buffer, ", "));
    //     throw std::invalid_argument(
    //   "Could not find device with this name! Available devices: [" +
    //   buffer.str() + ']');
    }
}

Device Devices::findDevice(char* uuid){
    //TODO
}

std::vector<Device> Devices::filter(bool (*con)(Device device)){
    std::vector<Device> filterd;
    std::copy_if (m_devices.begin(), m_devices.end(), std::back_inserter(filterd), con);
    return filterd;
}