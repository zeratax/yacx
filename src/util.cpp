#include "../include/cudaexecutor/util.hpp"

std::string cudaexecutor::load(const std::string &path) {
  std::ifstream file(path);
  return std::string((std::istreambuf_iterator<char>(file)),
                     std::istreambuf_iterator<char>());
}
template <typename T>
std::string
cudaexecutor::to_comma_separated(const std::vector<T> &vector) {
  std::string result;
  if (!vector.empty()) {
    for (const auto &i : vector) {
      result.append(std::to_string(i));
      result.append(", ");
    }
    result.substr(0, result.length() - 2);
  }
  return result;
}