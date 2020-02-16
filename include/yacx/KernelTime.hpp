#pragma once
#include "Devices.hpp"
#include "KernelArgs.hpp"
namespace yacx {
typedef struct {
  float upload{0};
  float download{0};
  float launch{0};
  float total{0};
} KernelTime;

float effective_bandwidth(float miliseconds, KernelArgs args);
float theoretical_bandwidth(Device* device);
} // namespace yacx
