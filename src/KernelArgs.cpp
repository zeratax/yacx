#include "yacx/KernelArgs.hpp"

#include <algorithm>

using yacx::KernelArgs, yacx::KernelArg, yacx::arg_type;

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

float KernelArgs::download(void *hdata) {
  float result{0};
  for (auto &arg : m_args)
    result += arg.download(hdata);
  return result;
}

const void **KernelArgs::content() {
  m_voArgs.resize(m_args.size());
  std::transform(m_args.begin(), m_args.end(), m_voArgs.begin(),
                 [](auto &arg) { return arg.content(); });
  return m_voArgs.data();
};

size_t KernelArgs::size(type) const {
  size_t result{0};
  for (auto const &arg : m_args) {
    if (arg.m_copy && (type == arg_type::UPLOAD || type == arg_type::TOTAL))
      result += arg.size();
    if (arg.m_download && (type == arg_type::DOWNLOAD || type == arg_type::TOTAL))
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
