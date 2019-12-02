#include "cudaexecutor/Kernel.hpp"
#include "cudaexecutor/Exception.hpp"
#include <utility>

using cudaexecutor::Kernel, cudaexecutor::loglevel;

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

Kernel &Kernel::launch(std::vector<KernelArg> args, Device device) {
  logger(loglevel::DEBUG) << "creating context and loading module";

  CUDA_SAFE_CALL(cuCtxCreate(&m_context, 0, device.get()));
  logger(loglevel::DEBUG1) << m_ptx.get();
  CUDA_SAFE_CALL(cuModuleLoadDataEx(&m_module, m_ptx.get(), 0, nullptr, nullptr));

  logger(loglevel::DEBUG) << "uploading arguments";
  const void *kernel_args[args.size()];
  int i{0};
  for (auto &arg : args) {
    arg.upload();
    kernel_args[i++] = arg.content();
  }
  logger(loglevel::DEBUG) << "getting function for " << m_demangled_name.c_str();
  CUDA_SAFE_CALL(
      cuModuleGetFunction(&m_kernel, m_module, m_demangled_name.c_str()));

  logger(loglevel::INFO) << "launching " << m_demangled_name;
  //CUDA_SAFE_CALL(cuCtxSynchronize());
  CUDA_SAFE_CALL(cuLaunchKernel(m_kernel, // function from program
                                m_grid.x, m_grid.y, m_grid.z,    // grid dim
                                m_block.x, m_block.y, m_block.z, // block dim
                                0, nullptr, // shared mem and stream
                                const_cast<void **>(kernel_args), // arguments
                                nullptr));
  CUDA_SAFE_CALL(cuCtxSynchronize());
  logger(loglevel::INFO) << "done!";

  // download results to host
  logger(loglevel::DEBUG) << "downloading arguments";
  for (auto &arg : args)
    arg.download();

  logger(loglevel::DEBUG) << "freeing resources";
  CUDA_SAFE_CALL(cuModuleUnload(m_module));
  CUDA_SAFE_CALL(cuCtxDestroy(m_context));

  return *this;
}
