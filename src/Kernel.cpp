#include "yacx/Kernel.hpp"
#include "yacx/Exception.hpp"
#include "yacx/Init.hpp"
#include "yacx/KernelArgs.hpp"
#include "yacx/KernelTime.hpp"
#include <utility>

using yacx::Kernel, yacx::KernelTime, yacx::loglevel, yacx::arg_type, yacx::eventInterval;

Kernel::Kernel(std::shared_ptr<char[]> ptx, std::string demangled_name)
    : m_ptx{std::move(ptx)}, m_demangled_name{std::move(demangled_name)} {
  Logger(loglevel::DEBUG) << "created templated Kernel " << m_demangled_name;

  Logger(loglevel::DEBUG1) << m_ptx.get();

  m_kernelFunction =
      std::make_shared<struct KernelFunction>(m_ptx.get(), m_demangled_name);
}

Kernel::KernelFunction::KernelFunction(char *ptx, std::string demangled_name) {
  yacx::detail::initCtx();

  Logger(loglevel::DEBUG) << "loading module";
  CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, ptx, 0, nullptr, nullptr));

  Logger(loglevel::DEBUG) << "getting function for " << demangled_name.c_str();
  CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, demangled_name.c_str()));
}

Kernel::KernelFunction::~KernelFunction() {
  Logger(loglevel::DEBUG) << "freeing module";

  CUDA_SAFE_CALL(cuModuleUnload(module));
}

Kernel &Kernel::configure(dim3 grid, dim3 block, unsigned int shared) {
  Logger(loglevel::DEBUG) << "configuring Kernel with grid: " << grid.x << ", "
                          << grid.y << ", " << grid.z << ", block: " << block.x
                          << ", " << block.y << ", " << block.z
                          << " and shared memory size: " << shared;
  m_grid = grid;
  m_block = block;
  m_shared = shared;

  if (m_shared > 0) {
    Logger(loglevel::DEBUG)
        << "kernel use " << shared << " dynamic shared memory";
    CUDA_SAFE_CALL(cuFuncSetAttribute(
        m_kernelFunction.get()->kernel,
        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, m_shared));
  }

  return *this;
}

float yacx::eventInterval::elapsed() {
  float time;

  CUDA_SAFE_CALL(cuEventElapsedTime(&time, start, end));

  return time;
}

eventInterval Kernel::asyncOperation(
    KernelArgs &args, CUstream stream, CUevent syncEvent,
    std::function<void(KernelArgs &args, CUstream stream)> operation) {
  CUevent start, end;
  CUDA_SAFE_CALL(cuEventCreate(&start, CU_EVENT_DEFAULT));
  CUDA_SAFE_CALL(cuEventCreate(&end, CU_EVENT_DEFAULT));

  if (syncEvent) {
    CUDA_SAFE_CALL(cuStreamWaitEvent(stream, syncEvent, 0));
  }

  CUDA_SAFE_CALL(cuEventRecord(start, stream));
  operation(args, stream);
  CUDA_SAFE_CALL(cuEventRecord(end, stream));

  return eventInterval{start, end};
}

eventInterval Kernel::uploadAsync(KernelArgs &args, Device &device,
                                  CUevent syncEvent) {
  Logger(loglevel::DEBUG) << "uploading arguments";

  return asyncOperation(
      args, device.getUploadStream(), syncEvent,
      [](KernelArgs &args, CUstream stream) { args.uploadAsync(stream); });
}

eventInterval Kernel::runAsync(KernelArgs &args, Device &device,
                               CUevent syncEvent) {
  Logger(loglevel::INFO) << "launching " << m_demangled_name;

  return asyncOperation(
      args, device.getLaunchStream(), syncEvent,
      [this](KernelArgs &args, CUstream stream) {
        CUDA_SAFE_CALL(cuLaunchKernel(
            m_kernelFunction.get()->kernel,  // function from program
            m_grid.x, m_grid.y, m_grid.z,    // grid dim
            m_block.x, m_block.y, m_block.z, // block dim
            m_shared, stream,                // dynamic shared mem and stream
            const_cast<void **>(args.content()), // arguments
            nullptr));
      });
}

eventInterval Kernel::downloadAsync(KernelArgs &args, Device &device,
                                    CUevent syncEvent, void *downloadDest) {
  Logger(loglevel::DEBUG) << "downloading arguments";

  return asyncOperation(args, device.getDownloadStream(), syncEvent,
                        [downloadDest](KernelArgs &args, CUstream stream) {
                          if (!downloadDest)
                            args.downloadAsync(stream);
                          else
                            args.downloadAsync(downloadDest, stream);
                        });
}

KernelTime Kernel::launch(KernelArgs args, Device &device) {
  Logger(loglevel::DEBUG) << "setting context";
  CUDA_SAFE_CALL(cuCtxSetCurrent(device.getPrimaryContext()));

  KernelTime time;
  CUevent start, stop;
  eventInterval upload, launch, download;

  CUDA_SAFE_CALL(cuEventCreate(&start, CU_EVENT_DEFAULT));
  CUDA_SAFE_CALL(cuEventCreate(&stop, CU_EVENT_DEFAULT));

  CUDA_SAFE_CALL(cuEventRecord(start, NULL));

  args.malloc();
  upload = uploadAsync(args, device, NULL);
  launch = runAsync(args, device, upload.end);
  download = downloadAsync(args, device, launch.end, NULL);

  time.size_upload = args.size(arg_type::UPLOAD);
  time.size_download = args.size(arg_type::DOWNLOAD);
  time.size_total = args.size(arg_type::TOTAL);

  CUDA_SAFE_CALL(cuStreamSynchronize(device.getDownloadStream()));
  args.free();

  CUDA_SAFE_CALL(cuEventRecord(stop, NULL));
  CUDA_SAFE_CALL(cuEventSynchronize(stop));

  time.upload = upload.elapsed();
  time.launch = launch.elapsed();
  time.download = download.elapsed();
  CUDA_SAFE_CALL(cuEventElapsedTime(&time.total, start, stop));

  return time;
}

std::vector<KernelTime>
Kernel::benchmark(KernelArgs args, unsigned int executions, Device &device) {
  Logger(loglevel::DEBUG) << "benchmarking kernel";

  std::vector<KernelTime> kernelTimes;
  kernelTimes.reserve(executions);

  std::vector<std::array<eventInterval, 3>> events;
  events.resize(executions);

  // find a kernelArg that you have to download with maximum size
  size_t maxOutputSize = args.maxOutputSize();

  Logger(loglevel::DEBUG) << "setting context";
  CUDA_SAFE_CALL(cuCtxSetCurrent(device.getPrimaryContext()));

  // allocate memory for output
  void *output;
  if (maxOutputSize) {
    CUDA_SAFE_CALL(cuMemAllocHost(&output, maxOutputSize));
  }

  Logger(loglevel::DEBUG) << "launch kernel " << executions << " times";

  args.malloc();

  events[0][0] = uploadAsync(args, device, NULL);
  events[0][1] = runAsync(args, device, events[0][0].end);
  // download results into output-memory (do not override input for next
  // execution)
  events[0][2] = downloadAsync(args, device, events[0][1].end, output);

  for (unsigned int i = 1; i < executions; i++) {
    events[i][0] = uploadAsync(args, device, events[i-1][2].end);
    events[i][1] = runAsync(args, device, events[i][0].end);
    events[i][2] = downloadAsync(args, device, events[i][1].end, output);
  }

  CUDA_SAFE_CALL(cuStreamSynchronize(device.getDownloadStream()));

  args.free();

  for (unsigned int i = 0; i < executions; i++) {
    KernelTime time;
    time.upload = events[i][0].elapsed();
    time.launch = events[i][1].elapsed();
    time.download = events[i][2].elapsed();
    time.total = time.upload + time.launch + time.download;

    time.size_upload = args.size(arg_type::UPLOAD);
    time.size_download = args.size(arg_type::DOWNLOAD);
    time.size_total = args.size(arg_type::TOTAL);

    kernelTimes.push_back(time);
  }

  // free allocated page-locked memory
  if (maxOutputSize) {
    CUDA_SAFE_CALL(cuMemFreeHost(output));
  }

  return kernelTimes;
}
