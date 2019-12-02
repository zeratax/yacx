#include "cudaexecutor/Headers.hpp"

#include <algorithm>
#include <cstdio>

using cudaexecutor::Header, cudaexecutor::Headers;

const char **Headers::content() const {
  m_chHeaders.resize(m_headers.size());
  std::transform(m_headers.begin(), m_headers.end(), m_chHeaders.begin(),
                 [](const auto &s) { return s.content(); });
  return m_chHeaders.data();
}

const char **Headers::names() const {
  m_chHeaders.resize(m_headers.size());
  std::transform(m_headers.begin(), m_headers.end(), m_chHeaders.begin(),
                 [](const auto &s) { return s.name(); });
  return m_chHeaders.data();
}

Headers::Headers(const Header &header) { insert(header); }

Headers::Headers(const std::string &path) { insert(path); }
