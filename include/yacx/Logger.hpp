#pragma once

#include "Colors.hpp"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

// we don't have P1275 but this works well enough as a substitute for now
namespace std {
std::vector<std::string const> arguments;
}

namespace yacx {
enum class loglevel { NONE, ERROR, WARNING, INFO, DEBUG, DEBUG1 };

using logmap = std::map<loglevel, std::pair<const char *, const char *>>;

logmap state = {
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

namespace detail {

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
  if (starts_with(flag, "-v") || starts_with(flag, "--verbose")) {
    Logger::getInstance().set_log_level(flag_value(flag));
  } else if (starts_with(flag, "-f") || starts_with(flag, "--file")) {
    Logger::getInstance().set_log_file(flag_value(flag));
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

static std::string get_datetime() {
  auto t = std::time(nullptr);
  auto tm = *std::localtime(&t);
  std::ostringstream oss;
  oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
  return oss.str();
}

} // namespace detail

void handle_logging_args(int argc, char const *const *const argv) {
  for (int i = 1; i < argc; ++i) {
    std::arguments.push_back(argv[i]);
  }
  detail::handle_flags(std::arguments);
}

/*!
  \class logger Logger.hpp
  \brief Class to log events of varying severity.
*/
class Logger {

 public:
  //! returns the logger instance.
  static Logger &getInstance() {
    static Logger instance;
    return instance;
  }

  //!
  Logger() {
#ifdef LOG_FILE
    set_logfile(LOG_FILE);
#endif
#ifdef LOG_LEVEL
    limit = static_cast<loglevel>(LOG_LEVEL);
#endif
#ifdef LOG_COUT
    cout_flag = LOG_COUT;
#endif
#ifdef LOG_CERR
    cerr_flag = LOG_CERR;
#endif
  }

  Logger(Logger const &) = delete;

  Logger &operator=(Logger const &) = delete;

  //! set the loglimit
  //! \param limit new limit for logging.
  void set_loglimit(const loglevel limit) { this->limit = limit; }

  //! set/unset cout as a logging output.
  //! \param flag flag that determines if cout has to be logged to.
  void set_cout(bool flag) { this->cout_flag = flag; }

  //! set/unset cerr as a logging output.
  //! \param flag flag that determines if cerr has to be logged to.
  void set_cerr(bool flag) { this->cerr_flag = flag; }

  //! sets a logfile.
  //! \param file The filename of the new logfile.
  void set_logfile(const std::string &file) {
    logfile_stream = std::make_unique<std::ofstream>(
        std::ofstream{file, std::ofstream::out | std::ofstream::app});
  }

  //! Prints to all appropriate logging outputs.
  //! \param value value to be printed.
  template <typename T> void print(T const &value) {
    if (cout_flag)
      std::cout << value << gColorReset;
    if (cerr_flag)
      std::cerr << value << gColorReset;
    if (logfile_stream) {
      *logfile_stream << value;
      if (logfile_stream->fail()) {
        std::cout << "Couldn't write to logfile.";
      }
    }
  }

  //! Prints a prefix to all appropriate logging outputs.
  //! Should not be discarded to prevent malformed logging output.
  //! \param severity The severity of the current logging reason.
  //! \param src_file The source file where the logging request was issued.
  //! \param src_line The source line where the logging request was issued.
  [[nodiscard]] Logger &prepare(loglevel severity, std::string src_file,
                                int src_line) {
    using namespace detail;
    current_loglevel = severity;
    if (current_loglevel <= limit) {
      std::stringstream prefix_ss;
      prefix_ss << std::endl
                << detail::get_color(severity) << get_datetime() << " "
                << get_name(severity) << "[" << src_file << ":" << src_line
                << "]: ";
      std::string prefix = prefix_ss.str();
      print(prefix);
    }
    return *this;
  }

  template <typename T> Logger &operator<<(T const &value) {
    if (current_loglevel <= limit) {
      print(value);
    }
    return *this;
  }

 private:
  bool cout_flag = true;
  bool cerr_flag = false;
  loglevel limit = loglevel::WARNING;

  loglevel current_loglevel;
  std::unique_ptr<std::ofstream> logfile_stream;
};

/*!
  \class log_null_sink log_null_sink.hpp
  \brief Class to discard logging requests of insufficient severity.
*/
class log_null_sink {
 public:
  template <typename T> log_null_sink &operator<<(T const &value) {
    return *this;
  }
};

#ifdef NO_LOGGING
#define Logger(level) yacx::log_null_sink()
#else
#define Logger(level)                                                          \
  yacx::Logger::getInstance().prepare(level, __FILE__, __LINE__)
#endif
} // namespace yacx
