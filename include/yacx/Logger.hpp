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

const char *get_name(loglevel level);

const char *get_color(loglevel level);
std::string get_datetime();

} // namespace detail

void handle_logging_args(int argc, char const *const *const argv);

/*!
  \class Logger Logger.hpp
  \brief Class to log events of varying severity.
*/
class Logger {

 public:
  loglevel limit = loglevel::WARNING;

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
