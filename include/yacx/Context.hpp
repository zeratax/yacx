#pragma once

#include "Device.hpp"
#include <cuda.h>

namespace yacx {

class Context : JNIHandle {
  /*!
   \class Context Context.hpp
   \brief Class to manage the current CUDA context
  */
 public:
  Context(Device device = Device());
  [[nodiscard]] CUcontext get() { return m_context }

 private:
  CUcontext m_context;
}

} // namespace yacx
