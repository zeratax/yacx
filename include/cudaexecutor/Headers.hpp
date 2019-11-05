#ifndef CUDAEXECUTOR_HEADERS_HPP_
#define CUDAEXECUTOR_HEADERS_HPP_

#include <string>
#include <vector>

#include "util.hpp"

namespace cudaexecutor {

class Header {
  std::string _path{};
  std::string _content{};

public:
  explicit Header(const std::string &path)
      : _path{path}, _content{load(path)} {}
  const char *get_name() const { return _path.c_str(); }
  const char *get_content() const { return _content.c_str(); }
};

class Headers {
  std::vector<Header> headers;

public:
  const char **content() const;
  const char **names() const;
  int size() const { return headers.size(); }
  void insert(std::string const &path) { headers.push_back(Header(path)); }
  void insert(Header header) { headers.push_back(header); }
};

} // namespace cudaexecutor

#endif // CUDAEXECUTOR_HEADERS_HPP_
