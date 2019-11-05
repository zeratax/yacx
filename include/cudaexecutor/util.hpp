#ifndef CUDAEXECUTOR_UTIL_HPP_
#define CUDAEXECUTOR_UTIL_HPP_

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
namespace cudaexecutor {

std::string load(const std::string &path);
template <typename T>
std::string to_comma_separated(const std::vector<T> &vector);
template <typename T> std::string type_of(const T &variable);
} // namespace cudaexecutor

#endif // CUDAEXECUTOR_UTIL_HPP_
