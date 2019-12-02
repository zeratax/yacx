#include "cudaexecutor/Options.hpp"

#include <algorithm>

using cudaexecutor::Options;

void Options::insert(const std::string &op) { m_options.push_back(op); }

void Options::insert(const std::string &name, const std::string &value) {
  if (value.empty())
    Options::insert(name);
  else
    m_options.push_back(name + "=" + value);
}

const char **Options::options() const {
  m_chOptions.resize(m_options.size());
  std::transform(m_options.begin(), m_options.end(), m_chOptions.begin(),
                 [](const auto &s) { return s.c_str(); });
  return m_chOptions.data();
}
