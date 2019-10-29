#ifndef _HEADERS_H_
#define _HEADERS_H_

#include <string>
#include <vector>

#include "util.hpp"

namespace cudaexecutor {

class Header {
  std::string _path{};
  std::string _content{};

public:
  Header(const std::string &path) : _path{path}, _content{load(path)} {}
  const char *get_name() const { return _path.c_str(); }
  const char *get_content() const { return _content.c_str(); }
};

class Headers {
  std::vector<Header> headers;

public:
  const char **content() const;
  const char **names() const;
  int size() const { return headers.size(); };
  void insert(std::string const &path) { this->headers.push_back(path); }
};

} // namespace cudaexecutor

#endif // _HEADERS_H_