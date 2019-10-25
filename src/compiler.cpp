// "Copyright 2019 Jona Abdinghoff"
#include <nvrtc.h>
#include <cuda.h>
#include <stdio.h>
#include <fstream>
#include <boost/program_options.hpp>
namespace po = boost::program_options;

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

std::string load(const std::string& path) {
    std::ifstream file(path);
    return std::string((std::istreambuf_iterator<char>(file)),
      std::istreambuf_iterator<char>());
}

bool process_command_line(int argc, char** argv,
                          const std::string& kernel_path,
                          const std::vector<char*>& options,
                          const std::vector<char*>& headers) {
  int iport;
  try {
    po::options_description desc("Program Usage", 1024, 512);
    desc.add_options()
      ("help",     "produce help message")
      ("kernel,k",   po::value<std::string>(&kernel_path)->required(),
       "path to cuda kernel")
      ("options,o",   po::value<std::string>(&options)->multitoken(),
       "compile options")
      ("header,h",   po::value<std::vector<char*>(&headers)->multitoken(),
       "cuda kernel headers");

      po::variables_map vm;
      po::store(po::parse_command_line(argc, argv, desc), vm);

      if (vm.count("help")) {
        std::cout << desc << "\n";
        return false;
      }

      po::notify(vm);
    }
    catch(std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return false;
    }
    catch(...) {
        std::cerr << "Unknown error!" << "\n";
        return false;
    }

    std::stringstream ss;
    ss << iport;
    port = ss.str();

    return true;
}

int main(int argc, char** argv) {
  std::string* kernel_path;
  std::vector<char*> options;
  std::vector<char*> headers;

  bool result = process_command_line(argc, argv, kernel_path, options, headers);
  if (!result)
      return 1;

  char* kernel_string = load(kernel_path);

  // Create an instance of nvrtcProgram with the kernel string.
  nvrtcProgram prog;
  NVRTC_SAFE_CALL(
    nvrtcCreateProgram(&prog,           // prog
                       kernel_string,   // buffer
                       "kernel.cu",     // name
                       headers.size(),  // numHeaders
                       NULL,            // headers
                       &headers[0]));   // includeNames
  // Compile the program for compute_30 with fmad disabled.
  nvrtcResult compileResult = nvrtcCompileProgram(prog,          // prog
                                                  2,             // numOptions
                                                  &options[0]);  // options
  // Obtain compilation log from the program.
  size_t logSize;
  NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize));
  char *log = new char[logSize];
  NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log));
  std::cout << log << '\n';
  delete[] log;
  if (compileResult != NVRTC_SUCCESS) {
    exit(1);
  }
  // Obtain PTX from the program.
  size_t ptxSize;
  NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));
  char *ptx = new char[ptxSize];
  NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptx));
  // Destroy the program.
  NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));

  return 0;
}
