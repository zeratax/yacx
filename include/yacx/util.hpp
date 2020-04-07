#pragma once

#include "Exception.hpp"

#include <cstdio>
#include <cxxabi.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <typeinfo> // operator typeid
#include <vector>

namespace yacx {

void debug(const std::string &message);

std::string load(const std::string &path);

template <typename T> std::string type_of(const T &variable);

template <typename T>
struct is_string
    : public std::disjunction<
          std::is_arithmetic<typename std::decay<T>::type>,
          std::is_same<char *, typename std::decay<T>::type>,
          std::is_same<const char *, typename std::decay<T>::type>,
          std::is_same<std::string, typename std::decay<T>::type>> {};

template <typename T> std::string type_of(const T &) {
  std::string type_name;
  NVRTC_SAFE_CALL(nvrtcGetTypeName<T>(&type_name));
  return type_name;
}

} // namespace yacx
