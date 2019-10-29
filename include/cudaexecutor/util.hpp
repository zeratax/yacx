#ifndef _UTIL_H_
#define _UTIL_H_

#include <string>
//#include <fstream>
#include <iostream>
namespace cudaexecutor {

std::string load(const std::string &path);
std::string to_comma_separated(const std::vector<std::string> &vector);

} // namespace cudaexecutor

#endif // _UTIL_H_