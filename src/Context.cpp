#include <yacx/Context.hpp>
#include <yacx/Exception.hpp>
#include <yacx/Logger.hpp>

using yacx::Context, yacx::loglevel;

Context::Context(Device device) {
  logger(loglevel::DEBUG) << "create context for device " << device.name();
  CUDA_SAFE_CALL(cuCtxCreate(&m_context, 0, device.get()));

  unsigned int version;
  CUDA_SAFE_CALL(cuCtxGetApiVersion(m_context, &version));
  logger(loglevel::DEBUG1) << "context API version: " << version;
}

Context::~Context() {
  logger(loglevel::DEBUG) << "destroying context";
  CUresult result = cuCtxDestroy(m_context);
  if (result != CUDA_SUCCESS) {
    logger(loglevel::ERROR) << yacx::detail::whichError(result);
  }
}
