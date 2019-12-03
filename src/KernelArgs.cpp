#include "cudaexecutor/KernelArgs.hpp"

using cudaexecutor::KernelArgs, cudaexecutor::KernelArg;

KernelArgs::KernelArgs(std::vector<KernelArg> args) : m_args{args} {}

float KernelArgs::upload() {
  float result{0};
  for (auto &arg : m_args)
    result += arg.upload();
  return result
}

float KernelArgs::download() {
  float result{0};
  for (auto &arg : m_args)
    result += arg.download();
  return result
}

void **KernelArgs::content() {
  m_chArgs.resize(m_args.size());
  std::transform(m_args.begin(), m_args.end(), m_chArgs.begin(),
                 [](auto &arg) { return arg.content(); });
  return m_chArgs.data();
};