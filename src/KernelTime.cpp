#include "yacx/KernelTime.hpp"

namespace yacx {

float effective_bandwidth(float miliseconds, KernelArgs args) {
  return args.size() / miliseconds / 1e6;
}

float theoretical_bandwidth(Device &device) {
  return (device.memory_clock_rate() * 1e3 * (device.bus_width() / 8) * 2) /
         1e9;
}
} // namespace yacx