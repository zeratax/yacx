#include "yacx/Kernel.hpp"
#include "yacx/KernelTime.hpp"
#include "yacx/Exception.hpp"
#include "yacx/KernelArgs.hpp"
#include "yacx/KernelTime.hpp"
#include <utility>
#include <builtin_types.h>

using yacx::Kernel, yacx::KernelTime, yacx::loglevel;

Kernel::Kernel(std::shared_ptr<char[]> ptx, std::string demangled_name)
    : m_ptx{std::move(ptx)}, m_demangled_name{std::move(demangled_name)} {
  logger(loglevel::DEBUG) << "created templated Kernel " << m_demangled_name;
}

Kernel &Kernel::configure(dim3 grid, dim3 block) {
  logger(loglevel::DEBUG) << "configuring Kernel with grid: " << grid.x << ", "
                          << grid.y << ", " << grid.z << " and block "
                          << block.x << ", " << block.y << ", " << block.z;
  m_grid = grid;
  m_block = block;
  return *this;
}

KernelTime Kernel::launch(KernelArgs args, Device device) {
  KernelTime time;
  cudaEvent_t start, stop;

  logger(loglevel::DEBUG) << "creating context and loading module";

  CUDA_SAFE_CALL(cuCtxCreate(&m_context, 0, device.get()));
  CUDA_SAFE_CALL(cuEventCreate(&start, CU_EVENT_DEFAULT));
  CUDA_SAFE_CALL(cuEventCreate(&stop, CU_EVENT_DEFAULT));

  logger(loglevel::DEBUG1) << m_ptx.get();
  CUDA_SAFE_CALL(
      cuModuleLoadDataEx(&m_module, m_ptx.get(), 0, nullptr, nullptr));

  logger(loglevel::DEBUG) << "uploading arguments";
  time.upload = args.upload();
  logger(loglevel::DEBUG) << "getting function for "
                          << m_demangled_name.c_str();
  CUDA_SAFE_CALL(
      cuModuleGetFunction(&m_kernel, m_module, m_demangled_name.c_str()));

  logger(loglevel::INFO) << "launching " << m_demangled_name;

  CUDA_SAFE_CALL(cuEventRecord(start, 0));
  CUDA_SAFE_CALL(
      cuLaunchKernel(m_kernel,                        // function from program
                     m_grid.x, m_grid.y, m_grid.z,    // grid dim
                     m_block.x, m_block.y, m_block.z, // block dim
                     0, nullptr,                      // shared mem and stream
                     const_cast<void **>(args.content()), // arguments
                     nullptr));
  CUDA_SAFE_CALL(cuEventRecord(stop, 0));
  // CUDA_SAFE_CALL(cuCtxSynchronize());
  logger(loglevel::INFO) << "done!";

  // download results to host
  logger(loglevel::DEBUG) << "downloading arguments";
  time.download = args.download();

  CUDA_SAFE_CALL(cuEventSynchronize(stop));
  CUDA_SAFE_CALL(cuEventElapsedTime(&time.launch, start, stop));

  logger(loglevel::DEBUG) << "freeing resources";
  CUDA_SAFE_CALL(cuModuleUnload(m_module));
  CUDA_SAFE_CALL(cuCtxDestroy(m_context));

  time.sum = time.launch + time.download + time.upload;

  return time;
}
