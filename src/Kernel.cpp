#include "yacx/Kernel.hpp"
#include "yacx/Exception.hpp"
#include "yacx/Init.hpp"
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

KernelTime Kernel::launch(KernelArgs args, Device &device) {
  logger(loglevel::DEBUG) << "setting context";
  CUDA_SAFE_CALL(cuCtxSetCurrent(device.getPrimaryContext()));

  return launch(args, NULL);
}

KernelTime Kernel::launch(KernelArgs args, void *downloadDest) {
  KernelTime time;
  cudaEvent_t start, launch, finish, stop;

  logger(loglevel::DEBUG) << "loading module";

  CUstream stream;
  CUDA_SAFE_CALL(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

  CUDA_SAFE_CALL(
      cuEventCreate(&start, CU_EVENT_DEFAULT)); // start of Kernel launch
  CUDA_SAFE_CALL(
      cuEventCreate(&launch, CU_EVENT_DEFAULT)); // start of Kernel execution
  CUDA_SAFE_CALL(
      cuEventCreate(&finish, CU_EVENT_DEFAULT)); // end of Kernel execution
  CUDA_SAFE_CALL(
      cuEventCreate(&stop, CU_EVENT_DEFAULT)); // end of Kernel launch

  CUDA_SAFE_CALL(cuEventRecord(start, stream));

  logger(loglevel::DEBUG1) << m_ptx.get();
  CUDA_SAFE_CALL(
      cuModuleLoadDataEx(&m_module, m_ptx.get(), 0, nullptr, nullptr));

  CUDA_SAFE_CALL(
      cuModuleGetFunction(&m_kernel, m_module, m_demangled_name.c_str()));

  logger(loglevel::DEBUG) << "uploading arguments";
  args.upload(stream);
  logger(loglevel::DEBUG) << "getting function for "
                          << m_demangled_name.c_str();

  logger(loglevel::INFO) << "launching " << m_demangled_name;

  CUDA_SAFE_CALL(cuEventRecord(launch, stream));
  CUDA_SAFE_CALL(
      cuLaunchKernel(m_kernel,                        // function from program
                     m_grid.x, m_grid.y, m_grid.z,    // grid dim
                     m_block.x, m_block.y, m_block.z, // block dim
                     0, stream,                      // shared mem and stream
                     const_cast<void **>(args.content()), // arguments
                     nullptr));
  CUDA_SAFE_CALL(cuEventRecord(finish, stream));

  logger(loglevel::INFO) << "done!";

  // download results to host
  logger(loglevel::DEBUG) << "downloading arguments";
  if (!downloadDest)
    args.download(stream);
  else
    args.download(downloadDest, stream);

  args.free(stream);

  CUDA_SAFE_CALL(cuEventRecord(stop, stream));
  CUDA_SAFE_CALL(cuEventSynchronize(stop));
  CUDA_SAFE_CALL(cuEventElapsedTime(&time.upload, start, launch));
  CUDA_SAFE_CALL(cuEventElapsedTime(&time.launch, launch, finish));
  CUDA_SAFE_CALL(cuEventElapsedTime(&time.download, start, launch));
  CUDA_SAFE_CALL(cuEventElapsedTime(&time.total, start, stop));

  CUDA_SAFE_CALL(cuStreamDestroy(stream));

  logger(loglevel::DEBUG) << "freeing module";

  CUDA_SAFE_CALL(cuModuleUnload(m_module));

  return time;
}

std::vector<KernelTime>
Kernel::benchmark(KernelArgs args, unsigned int executions, Device &device) {
  logger(loglevel::DEBUG) << "benchmarking kernel";

  std::vector<KernelTime> kernelTimes;
  kernelTimes.reserve(executions);

  // find a kernelArg that you have to download with maximum size
  size_t maxOutputSize = args.maxOutputSize();

  logger(loglevel::DEBUG) << "setting context";
  CUDA_SAFE_CALL(cuCtxSetCurrent(device.getPrimaryContext()));

  // allocate memory
  void *output;
  if (maxOutputSize) {
    CUDA_SAFE_CALL(cuMemAllocHost(&output, maxOutputSize));
  }

  logger(loglevel::DEBUG) << "launch kernel " << executions << " times";

  for (unsigned int i = 0; i < executions; i++) {
    // launch kernel, but download results into output-memory (do not override
    // input for next execution)
    KernelTime kernelTime = launch(args, output);

    kernelTimes.push_back(kernelTime);
  }

  // free allocated page-locked memory
  if (maxOutputSize) {
    CUDA_SAFE_CALL(cuMemFreeHost(output));
  }

  return kernelTimes;
}
