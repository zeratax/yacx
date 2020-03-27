#include "yacx/Kernel.hpp"
#include "yacx/Exception.hpp"
#include "yacx/Init.hpp"
#include "yacx/KernelArgs.hpp"
#include "yacx/KernelTime.hpp"
#include <utility>

using yacx::Kernel, yacx::KernelTime, yacx::loglevel, yacx::eventInterval;

Kernel::Kernel(std::shared_ptr<char[]> ptx, std::string demangled_name)
    : m_ptx{std::move(ptx)}, m_demangled_name{std::move(demangled_name)} {
  logger(loglevel::DEBUG) << "created templated Kernel " << m_demangled_name;

  yacx::detail::initCtx();

  logger(loglevel::DEBUG1) << m_ptx.get();
  logger(loglevel::DEBUG) << "loading module";
  CUDA_SAFE_CALL(
      cuModuleLoadDataEx(&m_module, m_ptx.get(), 0, nullptr, nullptr));

  logger(loglevel::DEBUG) << "getting function for "
                          << m_demangled_name.c_str();
  CUDA_SAFE_CALL(
      cuModuleGetFunction(&m_kernel, m_module, m_demangled_name.c_str()));
}

Kernel::~Kernel(){
  logger(loglevel::DEBUG) << "freeing module";

  CUDA_SAFE_CALL(cuModuleUnload(m_module));
}

Kernel &Kernel::configure(dim3 grid, dim3 block, unsigned int shared) {
  logger(loglevel::DEBUG) << "configuring Kernel with grid: " << grid.x << ", "
                          << grid.y << ", " << grid.z << ", block: " << block.x
                          << ", " << block.y << ", " << block.z
                          << " and shared memory size: " << shared;
  m_grid = grid;
  m_block = block;
  m_shared = shared;
  return *this;
}

float yacx::eventInterval::elapsed(){
  float time;
  
  CUDA_SAFE_CALL(cuEventElapsedTime(&time, start, end));

  return time;
}

eventInterval Kernel::asyncOperation(KernelArgs& args, CUstream stream, CUevent syncEvent,
    std::function<void (KernelArgs& args, CUstream stream)> operation){
  CUevent start, end;
  CUDA_SAFE_CALL(
      cuEventCreate(&start, CU_EVENT_DEFAULT));
  CUDA_SAFE_CALL(
      cuEventCreate(&end, CU_EVENT_DEFAULT));

  if (syncEvent){
    CUDA_SAFE_CALL(cuStreamWaitEvent(stream, syncEvent, 0));
  }

  CUDA_SAFE_CALL(cuEventRecord(start, stream));
  operation(args, stream);
  CUDA_SAFE_CALL(cuEventRecord(end, stream));

  return eventInterval{start, end};
}

eventInterval Kernel::uploadAsync(KernelArgs& args, Device& device){
  logger(loglevel::DEBUG) << "uploading arguments";

  return asyncOperation(args, device.getUploadStream(), NULL,
    [](KernelArgs& args, CUstream stream) {
      args.uploadAsync(stream);
  });
}

eventInterval Kernel::runAsync(KernelArgs& args, Device& device, CUevent syncEvent){
  logger(loglevel::INFO) << "launching " << m_demangled_name;
  
  return asyncOperation(args, device.getLaunchStream(), syncEvent,
    [this](KernelArgs& args, CUstream stream) {
      CUDA_SAFE_CALL(
      cuLaunchKernel(m_kernel,                            // function from program
                     m_grid.x, m_grid.y, m_grid.z,        // grid dim
                     m_block.x, m_block.y, m_block.z,     // block dim
                     m_shared, stream,                    // dynamic shared mem and stream
                     const_cast<void **>(args.content()), // arguments
                     nullptr));
                     logger(loglevel::INFO) << "end run operation " << m_demangled_name;
  });
}

eventInterval Kernel::downloadAsync(KernelArgs& args, Device& device, CUevent syncEvent, void *downloadDest){
  logger(loglevel::DEBUG) << "downloading arguments";

  return asyncOperation(args, device.getDownloadStream(), syncEvent,
    [downloadDest](KernelArgs& args, CUstream stream) {
        if (!downloadDest)
          args.downloadAsync(stream);
        else
          args.downloadAsync(downloadDest, stream);
  });
}

KernelTime Kernel::launch(KernelArgs args, Device &device) {
  logger(loglevel::DEBUG) << "setting context";
  CUDA_SAFE_CALL(cuCtxSetCurrent(device.getPrimaryContext()));

  if (m_shared > 0){
    logger(loglevel::DEBUG) << "kernel use " << m_shared << " dynamic shared memory";
    CUDA_SAFE_CALL(cuFuncSetAttribute(
        m_kernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, m_shared));
  }

  KernelTime time;
  CUevent start, stop;
  eventInterval upload, launch, download;

  CUDA_SAFE_CALL(
      cuEventCreate(&start, CU_EVENT_DEFAULT));
  CUDA_SAFE_CALL(
      cuEventCreate(&stop, CU_EVENT_DEFAULT));

  CUDA_SAFE_CALL(cuEventRecord(start, NULL));

  args.malloc();
  upload = uploadAsync(args, device);
  launch = runAsync(args, device, upload.end);
  download = downloadAsync(args, device, launch.end, NULL);
  CUDA_SAFE_CALL(cuStreamSynchronize(device.getDownloadStream()));
  args.free();

  CUDA_SAFE_CALL(cuEventRecord(stop, NULL));
  CUDA_SAFE_CALL(cuEventSynchronize(stop));

  logger(loglevel::DEBUG) << "upload";
  time.upload = upload.elapsed();
  logger(loglevel::DEBUG) << "launch";
  time.launch = launch.elapsed();
  logger(loglevel::DEBUG) << "download";
  time.download = download.elapsed();
  CUDA_SAFE_CALL(cuEventElapsedTime(&time.total, start, stop));

  return time;
}

std::vector<KernelTime>
Kernel::benchmark(std::vector<KernelArg>& argsVector, unsigned int executions, Device &device) {
  logger(loglevel::DEBUG) << "benchmarking kernel";

  std::vector<KernelTime> kernelTimes;
  kernelTimes.reserve(executions);

  std::vector<std::array<eventInterval, 3>> events;
  events.resize(executions);

  KernelArgs args[3] = {KernelArgs{argsVector}, KernelArgs{argsVector}, KernelArgs{argsVector}};

  // find a kernelArg that you have to download with maximum size
  size_t maxOutputSize = args[0].maxOutputSize();

  logger(loglevel::DEBUG) << "setting context";
  CUDA_SAFE_CALL(cuCtxSetCurrent(device.getPrimaryContext()));

  // allocate memory
  void *output;
  if (maxOutputSize) {
    CUDA_SAFE_CALL(cuMemAllocHost(&output, maxOutputSize));
  }

  logger(loglevel::DEBUG) << "launch kernel " << executions << " times";

  args[0].malloc();
  args[1].malloc();
  args[2].malloc();

  for (unsigned int i = 0; i < 3 && i < executions; i++) {
    events[i][0] = uploadAsync(args[i % 3], device);
    events[i][1] = runAsync(args[i % 3], device, events[i][0].end);
    // download results into output-memory (do not override input for next execution)
    events[i][2] = downloadAsync(args[i % 3], device, events[i][1].end, output);
  }

  for (unsigned int i = 3; i < executions; i++) {
    CUDA_SAFE_CALL(cuStreamWaitEvent(device.getUploadStream(), events[i-3][2].end, 0));

    events[i][0] = uploadAsync(args[i % 3], device);
    events[i][1] = runAsync(args[i % 3], device, events[i][0].end);
    // download results into output-memory (do not override input for next execution)
    events[i][2] = downloadAsync(args[i % 3], device, events[i][1].end, output);
  }

  CUDA_SAFE_CALL(cuStreamSynchronize(device.getDownloadStream()));

  args[0].free();
  args[1].free();
  args[2].free();

  for (unsigned int i = 0; i < executions; i++) {
    KernelTime time;
    time.upload = events[i][0].elapsed();
    time.launch = events[i][1].elapsed();
    time.download = events[i][2].elapsed();
    time.total = time.upload + time.launch + time.download;

    kernelTimes.push_back(time);
  }

  // free allocated page-locked memory
  if (maxOutputSize) {
    CUDA_SAFE_CALL(cuMemFreeHost(output));
  }

  return kernelTimes;
}
