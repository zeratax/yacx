#include "../include/cudaexecutor/Headers.hpp"

using cudaexecutor::Header, cudaexecutor::Headers;

const char **Headers::content() const {
  std::vector<const char *> result; // new?
  for (const auto &header : this->headers) {
    result.push_back(header.get_content());
  }
  return &result[0];
}

const char **Headers::names() const {
  std::vector<const char *> result; // new?
  for (const auto &name : this->headers) {
    result.push_back(name.get_name());
  }
  return &result[0];
}