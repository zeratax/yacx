#include "../include/yacx/Logger.hpp"
#include <unordered_map>

using yacx::Logger using yacx::loglevel;

// we don't have P1275 but this works well enough as a substitute for now
namespace std {
std::vector<std::string> arguments;
}

namespace yacx::detail {

static std::unordered_map<std::string, loglevel> const loglevels = {
    {"NONE", loglevel::NONE},       {"ERROR", loglevel::ERROR},
    {"WARNING", loglevel::WARNING}, {"INFO", loglevel::INFO},
    {"DEBUG", loglevel::DEBUG},   {"DEBUG1", loglevel::DEBUG1}};

// wishing for c++20
bool starts_with(const std::string &str, const std::string &substr) {
  return (str.rfind(substr, 0) == 0);
}

bool is_flag(const std::string &arg) { return arg[0] == '-'; }

std::string flag_value(const std::string &flag) {
  int pos = flag.find_first_of('=');
  if (pos != std::string::npos && pos < flag.size() - 1)
    return flag.substr(pos + 1, flag.size());
  throw std::invalid_argument("Value flags have the syntax --key=value");
}

void handle_flag(const std::string &flag) {
  if (starts_with(flag, "-l") || starts_with(flag, "--log")) {
    if (auto it = loglevels.find(flag); it != loglevels.end()) {
      Logger::getInstance().set_loglimit(
          static_cast<yacx::loglevel>(it->second));
    } else {
      std::invalid_argument(
          "only allowed log levels are ERROR, WARNING, INFO, DEBUG, DEBUG1");
    }
  } else if (starts_with(flag, "-f") || starts_with(flag, "--file")) {
    Logger::getInstance().set_logfile(flag_value(flag));
  }
}

void handle_flags(const std::vector<std::string> &flags) {
  for (auto &flag : flags) {
    if (is_flag(flag)) {
      handle_flag(flag);
    }
  }
}

const char *get_name(loglevel level) { return state.find(level)->second.first; }

const char *get_color(loglevel level) {
  return state.find(level)->second.second;
}

std::string get_datetime() {
  auto t = std::time(nullptr);
  auto tm = *std::localtime(&t);
  std::ostringstream oss;
  oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
  return oss.str();
}

} // namespace yacx::detail

yacx::logmap yacx::state = {
    {loglevel::NONE,
     std::pair<const char *, const char *>{"   NONE", gColorBrightDefault}},
    {loglevel::ERROR,
     std::pair<const char *, const char *>{"  ERROR", gColorBrightRed}},
    {loglevel::WARNING,
     std::pair<const char *, const char *>{"WARNING", gColorBrightYellow}},
    {loglevel::INFO,
     std::pair<const char *, const char *>{"   INFO", gColorBrightDefault}},
    {loglevel::DEBUG,
     std::pair<const char *, const char *>{"  DEBUG", gColorGray}},
    {loglevel::DEBUG1,
     std::pair<const char *, const char *>{" DEBUG1", gColorGray}}};

void yacx::handle_logging_args(int argc, char const *const *const argv) {
  for (int i = 1; i < argc; ++i) {
    std::arguments.push_back(argv[i]);
  }
  detail::handle_flags(std::arguments);
}