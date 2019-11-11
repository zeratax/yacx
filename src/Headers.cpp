#include "../include/cudaexecutor/Headers.hpp"

#include <algorithm>
#include <cstdio>

using cudaexecutor::Header, cudaexecutor::Headers;

const char **Headers::content() const {
  _chHeaders.resize(_headers.size());
  std::transform(_headers.begin(), _headers.end(), _chHeaders.begin(),
                 [](const auto &s) { return s.content(); });
  return _chHeaders.data();
}

const char **Headers::names() const {
  _chHeaders.resize(_headers.size());
  std::transform(_headers.begin(), _headers.end(), _chHeaders.begin(),
                 [](const auto &s) { return s.name(); });
  return _chHeaders.data();
}

Headers::Headers(const Header &header) { insert(header); }

Headers::Headers(const std::string &path) { insert(path); }
