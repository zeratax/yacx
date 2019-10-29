#ifndef _UTIL_H_
#define _UTIL_H_

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

#endif // _UTIL_H_