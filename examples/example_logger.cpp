#include "yacx/main.hpp"

using yacx::loglevel;

int main(int argc, char const *const *const argv) {
  yacx::handle_logging_args(argc, argv);

  Logger(loglevel::NONE) << "none message";
  Logger(loglevel::ERROR) << "error message";
  Logger(loglevel::WARNING) << "warning message";
  Logger(loglevel::INFO) << "info message";
  Logger(loglevel::DEBUG) << "debug message";
  Logger(loglevel::DEBUG1) << "debug message";

  return 0;
}