#include "../include/cudaexecutor/main.hpp"

using cudaexecutor::Program, cudaexecutor::ProgramArg, cudaexecutor::Kernel,
    cudaexecutor::Options, cudaexecutor::Device, cudaexecutor::load,
    cudaexecutor::type_of, cudaexecutor::to_comma_separated;

int main() {
  int array[]{5, 3, 3, 2, 7};
  int data{8};
  try {
    Device device;
    Options options{cudaexecutor::options::GpuArchitecture(device),
                    cudaexecutor::options::FMAD(false)};
    Program program{load("./examples/kernel/program.cu")};

    std::vector<ProgramArg> program_args;
    ProgramArg array_arg(&array, sizeof(int) * 5, true);
    ProgramArg data_arg(&data, sizeof(int));
    program_args.push_back(array_arg);
    program_args.push_back(data_arg);

    dim3 grid(1);
    dim3 block(1);
    program
        .kernel("my_kernel")           // returns Kernel type
        .instantiate(type_of(data), 5) // => Kernel
        .compile(options)              // => Kernel
        .configure(grid, block)        // => Kernel
        .launch(program_args);
  } catch (const cudaexecutor::cuda_exception &e) {
    std::cerr << e.what() << std::endl;
  } catch (const cudaexecutor::nvrtc_exception &e) {
    std::cerr << e.what() << std::endl;
  }

  std::vector<int> vec(array, array + 5);
  std::cout << to_comma_separated(vec) << std::endl;

  return 0;
}
