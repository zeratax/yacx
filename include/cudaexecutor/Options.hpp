#ifndef CUDAEXECUTOR_OPTIONS_HPP_
#define CUDAEXECUTOR_OPTIONS_HPP_

#include <string>
#include <vector>

#include "util.hpp"

#include <cuda.h>

namespace cudaexecutor {

class Option {};

class Options {
  std::vector<std::string> _options;
  mutable std::vector<const char *> _chOptions;

 public:
  void insert(const std::string &op);
  void insert(const std::string &name, const std::string &value) const;
  char **options() const;
  auto numOptions() const { return _options.size(); }
};

namespace detail {

class BooleanOption {
  const bool _b;

 public:
  explicit BooleanOption(bool b) : _b{b} {}
  auto value() const { return (_b) ? "true" : "false"; }
};

} // namespace detail

namespace options {

class GpuArchitecture {
  const std::string arc;

 public:
  GpuArchitecture(int major, int minor)
      : arc(std::string("compute_") + std::to_string(major) +
            std::to_string(minor)) {}

  explicit GpuArchitecture(const CudaDeviceProp &prop)
      : GpuArchitecture(prop.major, prop.minor) {}

  auto name() const { return "--gpu-architecture"; }
  auto &value() const { return arc; }
};

class FMAD : public detail::BooleanOption {
 public:
  using detail::BooleanOption::BooleanOption;
  auto name() const { return "--fmad"; }
};

class Fast_Math : public detail::BooleanOption {
 public:
  using detail::BooleanOption::BooleanOption;
  auto name() const { return "--use_fast_math"; }
};

} // namespace options

} // namespace cudaexecutor

#endif // CUDAEXECUTOR_OPTIONS_HPP_