#pragma once

#include <bits/unique_ptr.h>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

namespace yacx {

enum class loglevel { NONE, ERROR, WARNING, INFO, DEBUG, DEBUG1 };

static std::string get_name(loglevel l) {
  switch (l) {
  case loglevel::NONE:
    return "   NONE";
  case loglevel::ERROR:
    return "  ERROR";
  case loglevel::WARNING:
    return "WARNING";
  case loglevel::INFO:
    return "   INFO";
  case loglevel::DEBUG:
    return "  DEBUG";
  case loglevel::DEBUG1:
    return " DEBUG1";
  default:
    return "UNKNOWN";
  }
}

static std::string get_datetime() {
  auto t = std::time(nullptr);
  auto tm = *std::localtime(&t);
  std::ostringstream oss;
  oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
  return oss.str();
}

/*!
  \class logger Logger.hpp
  \brief Class to log events of varying severity.
*/
class logger {

 public:
    //! returns the logger instance.
    static logger &getInstance() {
        static logger instance;
        return instance;
    }

  //!
  logger() {
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

  logger(logger const &) = delete;

  logger &operator=(logger const &) = delete;

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
      std::cout << value;
    if (cerr_flag)
      std::cerr << value;
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
  [[nodiscard]] logger &prepare(loglevel severity, std::string src_file,
                                int src_line) {
    current_loglevel = severity;
    if (current_loglevel <= limit) {
      std::stringstream prefix_ss;
      prefix_ss << std::endl
                << get_datetime() << " " << get_name(severity) << "["
                << src_file << ":" << src_line << "]: ";
      std::string prefix = prefix_ss.str();
      print(prefix);
    }
    return *this;
  }

  template <typename T> logger &operator<<(T const &value) {
    if (current_loglevel <= limit) {
      print(value);
    }
    return *this;
  }

 private:
  bool cout_flag = true;
  bool cerr_flag = false;
  loglevel limit = loglevel::DEBUG1;

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
>>>>>>> 1e84400111284fc2daa4d70fc1acbdf03438b03e

#ifdef NO_LOGGING
#define logger(level) yacx::log_null_sink()
#else
#define logger(level) yacx::logger::getInstance()prepare(level, __FILE__, __LINE__)
#endif
} // namespace yacx
