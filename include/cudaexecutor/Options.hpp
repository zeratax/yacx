#pragma once

#include <string>
#include <vector>

#include "Device.hpp"
#include "util.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

namespace cudaexecutor {

class Options {
  std::vector<std::string> _options;
  mutable std::vector<const char *> _chOptions;

 public:
  Options() {}
  template <typename T> Options(const T &t);
  template <typename T, typename... TS> Options(const T &t, const TS &... ts);
  void insert(const std::string &op);
  void insert(const std::string &name, const std::string &value);
  template <typename T> void insertOptions(const T &t);
  template <typename T, typename... TS>
  void insertOptions(const T &t, const TS &... ts);
  const char **options() const;
  auto numOptions() const { return _options.size(); }
};

template <typename T> Options::Options(const T &t) { insertOptions(t); }

template <typename T, typename... TS>
Options::Options(const T &t, const TS &... ts) {
  insertOptions(t);
  insertOptions(ts...);
}

template <typename T> void Options::insertOptions(const T &t) {
  insert(t.name(), t.value());
}

template <typename T, typename... TS>
void Options::insertOptions(const T &t, const TS &... ts) {
  insert(t.name(), t.value());
  insertOptions(ts...);
}

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

  explicit GpuArchitecture(const cudaexecutor::Device &device)
      : GpuArchitecture(device.major(), device.minor()) {}

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
