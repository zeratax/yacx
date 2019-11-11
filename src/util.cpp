#include "../include/cudaexecutor/util.hpp"

std::string cudaexecutor::load(const std::string &path) {
  std::ifstream file(path);
  return std::string((std::istreambuf_iterator<char>(file)),
                     std::istreambuf_iterator<char>());
}
