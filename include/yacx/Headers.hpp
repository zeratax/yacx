#pragma once

#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

#include "util.hpp"
#include "JNIHandle.hpp"

namespace yacx {
/*!
  \class Header Headers.hpp
  \brief Class to help import header files for Source
*/
class Header {
 public:
  //!
  //! \param path relative path to header file
  explicit Header(const std::string &path)
      : _path{path}, _content{load(path)} {}
  const char *name() const { return _path.c_str(); }
  size_t length() const { return _path.size(); }
  //!
  //! \return c-style string of header content
  const char *content() const { return _content.c_str(); }

 private:
  std::string _path{};
  std::string _content{};
};

//! checks if type is Header or can be casted to Header
//! \tparam T type
template <typename T>
struct is_header
    : public std::disjunction<
          std::is_same<char *, typename std::decay<T>::type>,
          std::is_same<const char *, typename std::decay<T>::type>,
          std::is_same<std::string, typename std::decay<T>::type>,
          std::is_same<yacx::Header, typename std::decay<T>::type>> {};

class Headers : JNIHandle {
  /*!
  \class Headers Headers.hpp
  \brief List of Header for Source
  \example docs/headers.cpp
  \example example_gauss.cpp
*/
 public:
  Headers() {}
  //! constructs Headers with Header
  //! \param header
  explicit Headers(const Header &header);
  //! constructs Headers with path to header file
  //! \param path path to header file
  explicit Headers(const std::string &path);
  //! constructs a header from a header vector
  //! \param headers
  explicit Headers(std::vector<Header> headers);
  //! constructs Headers with a multiple Header or paths to header files
  //! \tparam T Header, std::string or char[]
  //! \tparam TS Header, std::string or char[]
  //! \param arg Header or Path to header file
  //! \param args  Header or Path to header file
  template <typename T, typename... TS>
  Headers(const T &arg, const TS &... args);
  //!
  //! \return c-style string array of header file contents
  const char **content() const;
  //!
  //! \return c-style string array of header file names
  const char **names() const;
  //!
  //! \return number of header files
  size_t numHeaders() const { return m_headers.size(); }
  //! inserts Header
  //! \param path path to header file
  void insert(std::string const &path) { m_headers.push_back(Header(path)); }
  //! inserts Header
  //! \param header
  void insert(Header header) { m_headers.push_back(header); }

 private:
  std::vector<Header> m_headers;
  mutable std::vector<const char *> m_chHeaders;
};

template <typename T, typename... TS>
Headers::Headers(const T &arg, const TS &... args) : Headers{args...} {
  static_assert(is_header<T>::value,
                "must be yacx::header or std::string");
  insert(arg);
}

} // namespace yacx
