#include "yacx/Kernel.hpp"
#include "yacx/Exception.hpp"
#include "yacx/KernelArgs.hpp"
#include "yacx/KernelTime.hpp"
#include <builtin_types.h>
#include <utility>

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

KernelTime Kernel::launch(KernelArgs args, Context context) {
  KernelTime time;
  cudaEvent_t start, launch, finish, stop;

  logger(loglevel::DEBUG) << "set context";
  CUDA_SAFE_CALL(cuCtxSetCurrent(context.get()));

  CUDA_SAFE_CALL(
      cuEventCreate(&start, CU_EVENT_DEFAULT)); // start of Kernel launch
  CUDA_SAFE_CALL(
      cuEventCreate(&launch, CU_EVENT_DEFAULT)); // start of Kernel execution
  CUDA_SAFE_CALL(
      cuEventCreate(&finish, CU_EVENT_DEFAULT)); // end of Kernel execution
  CUDA_SAFE_CALL(
      cuEventCreate(&stop, CU_EVENT_DEFAULT)); // end of Kernel launch

  logger(loglevel::DEBUG) << "loading module";
  CUDA_SAFE_CALL(cuEventRecord(start, 0));

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

  CUDA_SAFE_CALL(cuEventRecord(launch, 0));
  CUDA_SAFE_CALL(
      cuLaunchKernel(m_kernel,                        // function from program
                     m_grid.x, m_grid.y, m_grid.z,    // grid dim
                     m_block.x, m_block.y, m_block.z, // block dim
                     0, nullptr,                      // shared mem and stream
                     const_cast<void **>(args.content()), // arguments
                     nullptr));
  CUDA_SAFE_CALL(cuEventRecord(finish, 0));
  // CUDA_SAFE_CALL(cuCtxSynchronize());
  logger(loglevel::INFO) << "done!";

  // download results to host
  logger(loglevel::DEBUG) << "downloading arguments";
  time.download = args.download();

  CUDA_SAFE_CALL(cuEventRecord(stop, 0));
  CUDA_SAFE_CALL(cuEventSynchronize(stop));
  CUDA_SAFE_CALL(cuEventElapsedTime(&time.launch, launch, finish));
  CUDA_SAFE_CALL(cuEventElapsedTime(&time.total, start, stop));

  logger(loglevel::DEBUG) << "freeing resources";
  CUDA_SAFE_CALL(cuModuleUnload(m_module));

  return time;
}
