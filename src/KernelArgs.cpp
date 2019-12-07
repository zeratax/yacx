#include "yacx/KernelArgs.hpp"

#include <algorithm>

using yacx::KernelArgs, yacx::KernelArg;

KernelArgs::KernelArgs(std::vector<KernelArg> args) : m_args{args} {}

float KernelArgs::upload() {
  float result{0};
  for (auto &arg : m_args)
    result += arg.upload();
  return result;
}

float KernelArgs::download() {
  float result{0};
  for (auto &arg : m_args)
    result += arg.download();
  return result;
}

const void **KernelArgs::content() {
  m_chArgs.resize(m_args.size());
  std::transform(m_args.begin(), m_args.end(), m_chArgs.begin(),
                 [](auto &arg) { return arg.content(); });
  return m_chArgs.data();
};

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