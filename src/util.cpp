#include "util.hpp"

using cudaexecutor::load, cudaexecutor::to_comma_separated;

std::string load(const std::string &path) {
  std::ifstream file(path);
  return std::string((std::istreambuf_iterator<char>(file)),
                     std::istreambuf_iterator<char>());
}

std::string to_comma_separated(const std::vector<std::string> &vector) {
  std::string result;
  if (!vector.empty()) {
    for (const auto &i : vector) {
      result.append(i);
      result.append(", ");
    }
    result.substr(0, result.length() - 2);
  }
  return result;
}