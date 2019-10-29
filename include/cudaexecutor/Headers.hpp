#ifndef _HEADERS_H_
#define _HEADERS_H_

#include <string>
#include <vector>

#include "util.hpp"

namespace cudaexecutor {

class Header {
  std::string path{};
  std::string content{};

public:
  Header(const std::string &path) : this->path{path}, content{load(path)} {}
  std::string name() const { return path; }
  std::string content() const { return content; }
};

class Headers {
  std::vector<Header> headers;

public:
  char **content() const;
  char **names() const;
  int size() const { return headers.size(); };
  void insert(std::string const &path) { this->headers.push_back(path); }
};

} // namespace cudaexecutor

#endif // _HEADERS_H_