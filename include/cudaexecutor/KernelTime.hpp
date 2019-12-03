#pragma once
#include "KernelArgs.hpp"
#include "Device.hpp"

typedef struct {
  float upload{0};
  float download{0};
  float launch{0};
  float sum{0};
} KernelTime;