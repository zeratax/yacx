#include "../include/cudaexecutor/Program.hpp"
#include "../include/cudaexecutor/Device.hpp"
#include "../include/cudaexecutor/Exception.hpp"
#include "../include/cudaexecutor/Kernel.hpp"
#include "../include/cudaexecutor/Options.hpp"
#include "../include/cudaexecutor/util.hpp"

#include <vector_types.h> // dim3

using cudaexecutor::Program, cudaexecutor::ProgramArg, cudaexecutor::Kernel,
    cudaexecutor::Options, cudaexecutor::load, cudaexecutor::type_of, cudaexecutor::to_comma_separated;

int main() {
  Device dev;
  Options options =
      Options({cudaexecutor::options::GpuArchitecture(device.properties()),
               cudaexecutor::options::FMAD(false)});
  Program program(load("./examples/kernel/program.cu"));

  std::vector<ProgramArg> program_args{};
  int[] array{5, 3, 3, 2, 7};
  int data{8};
  ProgramArg array_arg(&array, true);
  ProgramArg data_arg(&data);
  program_args.push_back(array_arg);
  program_args.push_back(data_arg);

  dim3 grid(1);
  dim3 block(1);
  try {
    program.kernel("my_kernel")
        .instantiate(type_of(data), 5)
        .compile(options)
        .configure(grid, block)
        .launch(program_args);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
  }

  std::vector<int> vec(array, array + 5);
  std::cout << to_comma_separated(vec) << std::endl;

  return 0;
}
