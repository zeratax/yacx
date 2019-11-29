#include "cudaexecutor/Options.hpp"

#include <algorithm>

using cudaexecutor::Options;

void Options::insert(const std::string &op) { _options.push_back(op); }

void Options::insert(const std::string &name, const std::string &value) {
  if (value.empty())
    Options::insert(name);
  else
    _options.push_back(name + "=" + value);
}

const char **Options::options() const {
  _chOptions.resize(_options.size());
  std::transform(_options.begin(), _options.end(), _chOptions.begin(),
                 [](const auto &s) { return s.c_str(); });
  return _chOptions.data();
}
