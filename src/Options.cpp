#include "../include/cudaexecutor/Options.hpp"

#include <algorithm>
#include <cstdio>

using cudaexecutor::Options;

template <typename T> Options::Options(const T &t) {
  _options.push_back(t.name(), t.value());
}

template <typename T, typename... TS>
Options::Options(const T &t, const TS &... ts) {
  _options.push_back(t.name(), t.value());
  Options::insertOptions(ts...);
}

void Options::insert(const std::string &op) { _options.push_back(op); }

void Options::insert(const std::string &name, const std::string &value) {
  if (value.empty())
    Options::insert(name);
  else
    _options.push_back(name + "=" + value);
}

template <typename T> void Options::insertOptions(const T &t) {
  _options.push_back(t.name(), t.value());
}

template <typename T, typename... TS>
void Options::insertOptions(const T &t, const TS &... ts) {
  _options.push_back(t.name(), t.value());
  Options::insertOptions(ts...);
}

const char **Options::options() const {
  _chOptions.resize(_options.size());
  std::transform(_options.begin(), _options.end(), _chOptions.begin(),
                 [](const auto &s) { return s.c_str(); });
  return _chOptions.data();
}
