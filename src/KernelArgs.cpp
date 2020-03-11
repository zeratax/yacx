#include "yacx/KernelArgs.hpp"
#include "yacx/Exception.hpp"

#include <cuda.h>
#include <algorithm>

using yacx::KernelArgs, yacx::KernelArg;

KernelArgs::KernelArgs(std::vector<KernelArg> args) : m_args{args} {}

void KernelArgs::upload(CUstream stream) {
  for (auto &arg : m_args)
    arg.upload(stream);
}

void KernelArgs::download(CUstream stream) {
  for (auto &arg : m_args)
    arg.download(stream);
}

void KernelArgs::download(void *hdata, CUstream stream) {
  for (auto &arg : m_args)
    arg.download(hdata, stream);
}

void KernelArgs::free(CUstream stream){
  CUDA_SAFE_CALL(cuStreamSynchronize(stream));

  for (auto &arg : m_args)
    arg.free();
}

const void **KernelArgs::content() {
  m_voArgs.resize(m_args.size());
  std::transform(m_args.begin(), m_args.end(), m_voArgs.begin(),
                 [](auto &arg) { return arg.content(); });
  return m_voArgs.data();
}

size_t KernelArgs::size() const {
  size_t result{0};
  for (auto const &arg : m_args) {
    if (arg.m_copy)
      result += arg.size();
    if (arg.m_download)
      result += arg.size();
  }
  return result;
}

size_t KernelArgs::maxOutputSize() const {
  size_t result{0};
  for (auto const &arg : m_args) {
    if (arg.m_download && arg.size() > result)
      result = arg.size();
  }
  return result;
}
