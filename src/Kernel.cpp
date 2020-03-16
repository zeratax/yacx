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

Kernel &Kernel::configure(dim3 grid, dim3 block, unsigned int shared) {
  logger(loglevel::DEBUG) << "configuring Kernel with grid: " << grid.x << ", "
                          << grid.y << ", " << grid.z << ", block: "
                          << block.x << ", " << block.y << ", " << block.z
                          << "and shared memory size: " << shared;
  m_grid = grid;
  m_block = block;
  m_shared = shared;
  return *this;
}

KernelTime Kernel::launch(KernelArgs args, Device device) {
  logger(loglevel::DEBUG) << "creating context";

  CUDA_SAFE_CALL(cuCtxCreate(&m_context, 0, device.get()));

  KernelTime time = launch(args, NULL);

  logger(loglevel::DEBUG) << "destroy context";

  CUDA_SAFE_CALL(cuCtxDestroy(m_context));

  return time;
}

KernelTime Kernel::launch(KernelArgs args, void *downloadDest) {
  KernelTime time;
  cudaEvent_t start, launch, finish, stop;

  logger(loglevel::DEBUG) << "loading module";

  CUDA_SAFE_CALL(
      cuEventCreate(&start, CU_EVENT_DEFAULT)); // start of Kernel launch
  CUDA_SAFE_CALL(
      cuEventCreate(&launch, CU_EVENT_DEFAULT)); // start of Kernel execution
  CUDA_SAFE_CALL(
      cuEventCreate(&finish, CU_EVENT_DEFAULT)); // end of Kernel execution
  CUDA_SAFE_CALL(
      cuEventCreate(&stop, CU_EVENT_DEFAULT)); // end of Kernel launch

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
  if (m_shared > 0) CUDA_SAFE_CALL(cuFuncSetAttribute(m_kernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, m_shared));
  CUDA_SAFE_CALL(
      cuLaunchKernel(m_kernel,                        // function from program
                     m_grid.x, m_grid.y, m_grid.z,    // grid dim
                     m_block.x, m_block.y, m_block.z, // block dim
                     m_shared, nullptr,               // shared mem and stream
                     const_cast<void **>(args.content()), // arguments
                     nullptr));
  CUDA_SAFE_CALL(cuEventRecord(finish, 0));
  // CUDA_SAFE_CALL(cuCtxSynchronize());
  logger(loglevel::INFO) << "done!";

  // download results to host
  logger(loglevel::DEBUG) << "downloading arguments";
  if (!downloadDest)
    time.download = args.download();
  else
    time.download = args.download(downloadDest);

  CUDA_SAFE_CALL(cuEventRecord(stop, 0));
  CUDA_SAFE_CALL(cuEventSynchronize(stop));
  CUDA_SAFE_CALL(cuEventElapsedTime(&time.launch, launch, finish));
  CUDA_SAFE_CALL(cuEventElapsedTime(&time.total, start, stop));

  logger(loglevel::DEBUG) << "freeing module";

  CUDA_SAFE_CALL(cuModuleUnload(m_module));

  return time;
}

std::vector<KernelTime>
Kernel::benchmark(KernelArgs args, unsigned int executions, Device device) {
  logger(loglevel::DEBUG) << "benchmarking kernel";

  std::vector<KernelTime> kernelTimes;
  kernelTimes.reserve(executions);

  // find a kernelArg that you have to download with maximum size
  size_t maxOutputSize = args.maxOutputSize();

  // allocate memory
  void *output;
  if (maxOutputSize) {
    output = malloc(maxOutputSize);
  }

  logger(loglevel::DEBUG) << "create context";

  // create context
  CUDA_SAFE_CALL(cuCtxCreate(&m_context, 0, device.get()));

  logger(loglevel::DEBUG) << "launch kernel " << executions << " times";

  for (unsigned int i = 0; i < executions; i++) {
    // launch kernel, but download results into output-memory (do not override
    // input for next execution)
    KernelTime kernelTime = launch(args, output);

    kernelTimes.push_back(kernelTime);
  }

  logger(loglevel::DEBUG) << "destroy context";

  // destroy context
  CUDA_SAFE_CALL(cuCtxDestroy(m_context));

  // free allocated memory
  if (maxOutputSize) {
    free(output);
  }

  return kernelTimes;
}
