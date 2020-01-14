#include <yacx/Context.hpp>
#include <yacx/Exception.hpp>
#include <yacx/Logger.hpp>

using yacx::Context, yacx::loglevel;

Context::Context(Device device) {
  logger(loglevel::DEBUG) << "create context for device " << device.name();
  CUDA_SAFE_CALL(cuCtxCreate(&m_context, 0, device.get()));
}

~Context::Context {
  logger(loglevel::DEBUG) << "destroying context";
  CUresult result = cuCtxDestroy(m_context);
  if (result != CUDA_SUCCESS) {
    // error handling
  }
}
