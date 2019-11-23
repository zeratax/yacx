#ifndef CUDAEXECUTOR_UTIL_HPP_
#define CUDAEXECUTOR_UTIL_HPP_

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

namespace cudaexecutor {

    void debug(const std::string &message);

    std::string load(const std::string &path);

    template<typename Iter>
    std::string to_comma_separated(Iter begin, Iter end, const std::string &separator = std::string{", "});

    template<typename T>
    std::string type_of(const T &variable);

// https://stackoverflow.com/a/57812868
    template<typename T>
    struct is_string
            : public std::disjunction<
                    std::is_arithmetic<typename std::decay<T>::type>,
                    std::is_same<char *, typename std::decay<T>::type>,
                    std::is_same<const char *, typename std::decay<T>::type>,
                    std::is_same<std::string, typename std::decay<T>::type>> {
    };

    template<typename Iter>
    std::string to_comma_separated(Iter begin, Iter end, const std::string &separator) {
        static_assert(is_string<typename std::iterator_traits<Iter>::value_type>::value, "vector element must be stringable");
        std::ostringstream oss;
        while (begin != end) {
            oss << *begin;
            ++begin;
            if (begin != end) oss << separator;
        }
        return oss.str();
    }

// template <typename T> std::string type_of(const T &variable) {
//  int status;
//  std::string tname = typeid(T).name();
//  char *demangled_name =
//      abi::__cxa_demangle(tname.c_str(), NULL, NULL, &status);
//  if (status == 0) {
//    tname = demangled_name;
//    std::free(demangled_name);
//  }
//  return tname;
//}

    template<typename T>
    std::string type_of(const T &variable) {
        std::string type_name;
        NVRTC_SAFE_CALL(nvrtcGetTypeName<T>(&type_name));
        return type_name;
    }

} // namespace cudaexecutor

#endif // CUDAEXECUTOR_UTIL_HPP_
