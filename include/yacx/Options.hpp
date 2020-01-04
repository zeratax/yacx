#pragma once

#include <string>
#include <vector>

#include "Device.hpp"
#include "JNIHandle.hpp"
#include "util.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

namespace yacx {

/*!
  \class Options Options.hpp
  \brief Options for compiling a Program
  \example docs/options_construct.cpp
*/
class Options : JNIHandle {
 public:
  //! empty Options constructor
  Options() {}
  //! construct Options with one Option
  //! \tparam T optiontype, e.g. FMAD
  //! \param t option
  template <typename T> Options(const T &t);
  //! construct Options with multiple Option
  //! \tparam T  optiontype, e.g. FMAD
  //! \tparam TS Option
  //! \param t  optiontype, e.g. FMAD
  //! \param ts Option
  template <typename T, typename... TS> Options(const T &t, const TS &... ts);
  //! insert Option
  //! \param op e.g. "--device-debug"
  void insert(const std::string &op);
  //! insert Option
  //! \param name e.g. "--fmad"
  //! \param value e.g. "false"
  void insert(const std::string &name, const std::string &value);
  //! insert Option
  //! \tparam T optiontype, e.g. FMAD
  //! \param t Option
  template <typename T> void insertOptions(const T &t);
  //! insert multiple Options with multiple Option
  //! \tparam T  optiontype, e.g. FMAD
  //! \tparam TS Option
  //! \param t  optiontype, e.g. FMAD
  //! \param ts Option
  template <typename T, typename... TS>
  void insertOptions(const T &t, const TS &... ts);
  //!
  //! \return c-style string array with all options
  const char **content() const;
  //!
  //! \return number of Options
  auto numOptions() const { return m_options.size(); }

 private:
  std::vector<std::string> m_options;
  mutable std::vector<const char *> m_chOptions;
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
 public:
  explicit BooleanOption(bool b) : m_b{b} {}
  auto value() const { return (m_b) ? "true" : "false"; }

 private:
  const bool m_b;
};

} // namespace detail

namespace options {

class GpuArchitecture {
 public:
  GpuArchitecture(int major, int minor)
      : m_arc(std::string("compute_") + std::to_string(major) +
              std::to_string(minor)) {}

  explicit GpuArchitecture(const yacx::Device &device)
      : GpuArchitecture(device.major_version(), device.minor_version()) {}

  auto name() const { return "--gpu-architecture"; }
  auto &value() const { return m_arc; }

 private:
  const std::string m_arc;
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

} // namespace yacx
