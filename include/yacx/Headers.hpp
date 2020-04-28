#pragma once

#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

#include "JNIHandle.hpp"
#include "util.hpp"

namespace yacx {
/*!
  \class Header Headers.hpp
  \brief Class to help import header files for Source
  this is only interesting if the headers are not available at runtime
*/
class Header {
 public:
  //!
  //! \param name relative name to header file
  explicit Header(const std::string &name,std::string const &content)
      : _name{name}, _content{content} {}
  const char *name() const { return _name.c_str(); }
  //!
  //! \return c-style string of header content
  const char *content() const { return _content.c_str(); }

 private:
  std::string _content{};
  std::string _name{};
};

//! checks if type is Header or can be casted to Header
//! \tparam T type
template <typename T>
struct is_header
    : public std::disjunction<
          std::is_same<std::pair<std::string, char>, typename std::decay<T>::type>,
          std::is_same<std::pair<std::string, const char *>, typename std::decay<T>::type>,
          std::is_same<std::pair<char, std::string>, typename std::decay<T>::type>,
          std::is_same<std::pair<const char *, std::string>, typename std::decay<T>::type>,
          std::is_same<std::pair<std::string, std::string>, typename std::decay<T>::type>,
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
  //! constructs Headers with name to header file
  //! \param name name to header file
  explicit Headers(const std::string &name);
  //! constructs a header from a header vector
  //! \param headers
  explicit Headers(std::vector<Header> headers);
  //! constructs Headers with a multiple Header or names to header files
  //! \tparam T Header, std::string or char[]
  //! \tparam TS Header, std::string or char[]
  //! \param arg Header or name to header file
  //! \param args  Header or name to header file
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
  //! \param name name to header file
  void insert(std::pair<std::string>, std::string> header) { m_headers.push_back(Header(header.first, header.second)); }
  //! inserts Header
  //! \param header
  void insert(Header header) { m_headers.push_back(header); }

 private:
  std::vector<Header> m_headers;
  mutable std::vector<const char *> m_chHeaders;
};

template <typename T, typename... TS>
Headers::Headers(const T &arg, const TS &... args) : Headers{args...} {
  static_assert(is_header<T>::value, "must be yacx::header or std::string");
  insert(arg);
}

} // namespace yacx
