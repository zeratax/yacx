// "Copyright 2019 Jona Abdinghoff"
// #include <nvrtc.h>
// #include <cuda.h>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <boost/program_options.hpp>
namespace po = boost::program_options;

#define HELP_WIDTH 1024
#define HELP_HEIGHT 512
/*
#define NUM_THREADS 128
#define NUM_BLOCKS 32
#define NVRTC_SAFE_CALL(x)                                        \
  do {                                                            \
    nvrtcResult result = x;                                       \
    if (result != NVRTC_SUCCESS) {                                \
      std::cerr << "\nerror: " #x " failed with error "           \
                << nvrtcGetErrorString(result) << '\n';           \
      exit(1);                                                    \
    }                                                             \
  } while (0)
#define CUDA_SAFE_CALL(x)                                         \
  do {                                                            \
    CUresult result = x;                                          \
    if (result != CUDA_SUCCESS) {                                 \
      const char *msg;                                            \
      cuGetErrorName(result, &msg);                               \
      std::cerr << "\nerror: " #x " failed with error "           \
                << msg << '\n';                                   \
      exit(1);                                                    \
    }                                                             \
  } while (0)
*/

std::string load(const std::string &path) {
  std::ifstream file(path);
  return std::string((std::istreambuf_iterator<char>(file)),
    std::istreambuf_iterator<char>());
}

std::string to_comma_separated(const std::vector<std::string>& vector) {
  std::string result;
  if (!vector.empty()) {
    for (const auto & i : vector) {
      result.append(i);
      result.append(", ");
    }
    result.erase(result.end()-2, result.end());
  }
  return result;
}

bool process_command_line(int argc, char** argv,
                          std::string *kernel_path,
                          std::vector<std::string> *options,
                          std::vector<std::string> *headers) {
  try {
    po::options_description desc("Program Usage", HELP_WIDTH, HELP_HEIGHT);
    desc.add_options()
      ("help",     "produce help message")
      ("kernel,k",  po::value(kernel_path)->required(),
       "path to cuda kernel")
      ("options,o", po::value(options)->multitoken(),
       "compile options")
      ("header,h",  po::value(headers)->multitoken(),
       "cuda kernel headers");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help") != 0) {
      std::cout << desc << "\n";
      return false;
    }

    po::notify(vm);
  } catch(std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return false;
  } catch(...) {
    std::cerr << "Unknown error!" << "\n";
    return false;
  }

  return true;
}

int main(int argc, char** argv) {
  std::string kernel_path;
  std::vector<std::string> options;
  std::vector<std::string> headers;

  bool result = process_command_line(argc,
                                     argv,
                                     &kernel_path,
                                     &options,
                                     &headers);
  if (!result) {
    return 1;
}

  std::string kernel_string = load(kernel_path);
  std::string includeNames = to_comma_separated(headers);
  std::string compileOptions = to_comma_separated(options);

  std::cout << "Kernel String: \n'''\n" << kernel_string.c_str()  << "'''\n";
  std::cout << "Kernel Path: "    << kernel_path    << "\n";
  std::cout << "numHeaders: "     << headers.size() << "\n";
  std::cout << "includeNames: "   << includeNames   << "\n";
  std::cout << "compileOptions: " << compileOptions << "\n";

  /*
  // Create an instance of nvrtcProgram with the kernel string.
  nvrtcProgram prog;
  NVRTC_SAFE_CALL(
    nvrtcCreateProgram(&prog,           // prog
                       kernel_string,   // buffer
                       "kernel.cu",     // name
                       headers.size(),  // numHeaders
                       NULL,            // headers
                       headers to char**));   // includeNames
  // Compile the program for compute_30 with fmad disabled.
  nvrtcResult compileResult = nvrtcCompileProgram(prog,                // prog
                                                  options.size(),      // numOptions
                                                  options to char**);  // options
  // Obtain compilation log from the program.
  size_t logSize;
  NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize));
  char *log = new char[logSize];
  NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log));
  std::cout << log << '\n';
  delete[] log;
  if (compileResult != NVRTC_SUCCESS)
    exit(1);

  // Obtain PTX from the program.
  size_t ptxSize;
  NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));
  char *ptx = new char[ptxSize];
  NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptx));
  // Destroy the program.
  NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));
  */
  return 0;
}
