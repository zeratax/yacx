#include "yacx/util.hpp"

#include <string>
#include <vector>
#include <filesystem>

std::string yacx::load(const std::filesystem::path &path) {
  const std::filesystem::path current_file = __FILE__;
  const auto current_dir = current_file.parent_path();
  const auto absolute_path = current_dir + "../" + path

  std::ifstream file(absolute_path);
  return std::string((std::istreambuf_iterator<char>(file)),
                     std::istreambuf_iterator<char>());
}