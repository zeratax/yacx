#pragma once

namespace yacx {
namespace detail {
//! <code>true</code> if the CUDA driver API was instantiated,
//! <code>false</code> otherwise
extern bool instantiated;
//! Initialize the CUDA driver API if not already done
void init();

extern bool instantiatedCtx;
//! Initialize and set a context to current thread if not already done
void initCtx();
} // namespace detail
} // namespace yacx
