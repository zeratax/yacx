#pragma once

#include "Device.hpp"
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

// TODO
static bool instantiatedCtx = false;
static void initCtx() {
  if (!instantiatedCtx) {
    Device device;
    CUcontext ctx;
    CUDA_SAFE_CALL(cuDevicePrimaryCtxRetain(&ctx, device.get()));
    CUDA_SAFE_CALL(cuCtxPushCurrent(ctx));
  }
}
} // namespace detail
} // namespace yacx
