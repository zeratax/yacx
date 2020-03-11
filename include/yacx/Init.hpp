#pragma once

#include "Devices.hpp"
#include "Exception.hpp"
#include <cuda.h>

namespace yacx {
namespace detail {
//! <code>true</code> if the CUDA driver API was instantiated,
//! <code>false</code> otherwise
static bool instantiated = false;
//! Initialize the CUDA driver API if not already done
static void init() {
  if (!instantiated) {
    CUDA_SAFE_CALL(cuInit(0));
  }
}
} // namespace detail
} // namespace yacx
