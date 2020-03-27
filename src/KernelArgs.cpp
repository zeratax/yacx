#include "yacx/KernelArgs.hpp"
#include "yacx/Exception.hpp"

#include <algorithm>
#include <cuda.h>

using yacx::KernelArgs, yacx::KernelArg;

KernelArgs::KernelArgs(std::vector<KernelArg> args) : m_args{args} {}

void KernelArgs::malloc() {
  for (auto &arg : m_args)
    arg.malloc();
}

void KernelArgs::uploadAsync(CUstream stream) {
  for (auto &arg : m_args)
    arg.uploadAsync(stream);
}

void KernelArgs::downloadAsync(CUstream stream) {
  for (auto &arg : m_args)
    arg.downloadAsync(stream);
}

void KernelArgs::downloadAsync(void *hdata, CUstream stream) {
  for (auto &arg : m_args)
    arg.downloadAsync(hdata, stream);
}

void KernelArgs::free() {
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
