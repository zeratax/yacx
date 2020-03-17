#include "yacx/Init.hpp"

#include "yacx/Devices.hpp"
#include "yacx/Exception.hpp"
#include <cuda.h>

using yacx::Device;

bool yacx::detail::instantiated = false;
//! Initialize the CUDA driver API if not already done
void yacx::detail::init() {
  if (!instantiated) {
    CUDA_SAFE_CALL(cuInit(0));
  }
}

bool yacx::detail::instantiatedCtx = false;
//! Initialize and set a context to current thread if not already done
void yacx::detail::initCtx() {
  if (!instantiatedCtx) {
    Device &device = Devices::findDevice();
    CUDA_SAFE_CALL(cuCtxSetCurrent(device.getPrimaryContext()))
    instantiatedCtx = true;
  }
}