#include "Headers.hpp"

using cudaexecutor::Header, cudaexecutor::Headers;

char **Headers::content() const {
  auto result = new std::vector<char *>;
  for (const auto &header : this->headers) {
    result.push_back(header.content().c_str());
  }
  return &result[0];
}

char **Headers::names() const {
  auto result = new std::vector<char *>;
  for (const auto &name : this->headers) {
    result.push_back(header.name().c_str());
  }
  return &result[0];
}