#pragma once

#include "util.hpp"
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

#include "util.hpp"

namespace cudaexecutor {

class Header {
  std::string _path{};
  std::string _content{};

 public:
  explicit Header(const std::string &path)
      : _path{path}, _content{load(path)} {}
  const char *name() const { return _path.c_str(); }
  size_t length() const { return _path.size(); }
  const char *content() const { return _content.c_str(); }
};

template <typename T>
struct is_header
    : public std::disjunction<
          std::is_same<char *, typename std::decay<T>::type>,
          std::is_same<const char *, typename std::decay<T>::type>,
          std::is_same<std::string, typename std::decay<T>::type>,
          std::is_same<cudaexecutor::Header, typename std::decay<T>::type>> {};

class Headers {
  std::vector<Header> _headers;
  mutable std::vector<const char *> _chHeaders;

 public:
  Headers() {}
  explicit Headers(const Header &header);
  explicit Headers(const std::string &path);
  template <typename T, typename... TS>
  Headers(const T &arg, const TS &... args);
  const char **content() const;
  const char **names() const;
  int size() const { return _headers.size(); }
  void insert(std::string const &path) { _headers.push_back(Header(path)); }
  void insert(Header header) { _headers.push_back(header); }
};

template <typename T, typename... TS>
Headers::Headers(const T &arg, const TS &... args) : Headers{args...} {
  static_assert(is_header<T>::value,
                "must be cudaexecutor::header or std::string");
  insert(arg);
}

} // namespace cudaexecutor

#endif // CUDAEXECUTOR_HEADERS_HPP_
