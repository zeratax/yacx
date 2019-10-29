#include <boost/program_options.hpp>
#include <cstdio>
#include <string>

#include "../include/cudaexecutor/util.hpp"

using cudaexecutor::load, cudaexecutor::to_comma_separated;

namespace po = boost::program_options;

bool process_command_line(int argc, char **argv, std::string *kernel_path,
                          std::vector<std::string> *options,
                          std::vector<std::string> *headers) {
  const int HELP_WIDTH{1024};
  const int HELP_HEIGHT{512};
  try {
    po::options_description desc("Program Usage", HELP_WIDTH, HELP_HEIGHT);
    desc.add_options()("help", "produce help message")(
        "kernel,k", po::value(kernel_path)->required(), "path to cuda kernel")(
        "options,o", po::value(options)->multitoken(), "compile options")(
        "header,h", po::value(headers)->multitoken(), "cuda kernel headers");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help") != 0) {
      std::cout << desc << "\n";
      return false;
    }

    po::notify(vm);
  } catch (std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return false;
  } catch (...) {
    std::cerr << "Unknown error!"
              << "\n";
    return false;
  }

  return true;
}

int main(int argc, char **argv) {
  std::string kernel_path;
  std::vector<std::string> options;
  std::vector<std::string> headers;

  bool result =
      process_command_line(argc, argv, &kernel_path, &options, &headers);
  if (!result) {
    return 1;
  }

  std::string kernel_string = load(kernel_path);
  std::string includeNames = to_comma_separated(headers);
  std::string compileOptions = to_comma_separated(options);

  std::cout << "Kernel String: \n'''\n" << kernel_string << "'''\n";
  std::cout << "Kernel Path: " << kernel_path << "\n";
  std::cout << "numHeaders: " << headers.size() << "\n";
  std::cout << "includeNames: " << includeNames << "\n";
  std::cout << "compileOptions: " << compileOptions << "\n";

  return 0;
}
