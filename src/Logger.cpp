#include "../include/yacx/Logger.hpp"

using yacx::Logger;

namespace yacx::detail {

bool starts_with(const std::string &str, const std::string &substr) {
  return (str.rfind(substr, 0) == 0);
}

bool is_flag(const std::string &arg) { return arg[0] == '-'; }

std::string flag_value(const std::string &flag) {
  int pos = flag.find_first_of('=');
  if (pos != std::string::npos)
    return flag.substr(0, pos);
  throw std::invalid_argument("Value flags have the syntax --key=value");
}

void handle_flag(const std::string &flag) {
  if (starts_with(flag, "-l") || starts_with(flag, "--log")) {
    int level = std::stoi(flag_value(flag));
    Logger::getInstance().set_loglimit(static_cast<yacx::loglevel>(level));
  } else if (starts_with(flag, "-f") || starts_with(flag, "--file")) {
    Logger::getInstance().set_logfile(flag_value(flag));
  }
}

void handle_flags(const std::vector<const std::string> &flags) {
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

}

void yacx::handle_logging_args(int argc, char const *const *const argv) {
  for (int i = 1; i < argc; ++i) {
    std::arguments.push_back(argv[i]);
  }
  detail::handle_flags(std::arguments);
}