#include "Program.hpp"
#include "Exception.hpp"
#include "Kernel.hpp"
#include "Device.hpp"

#include <cuda.h>

using cudaexecutor::Program, cudaexecutor::ProgramArg, cudaexecutor::Kernel,
    cudaexecutor::load, cudaexecutor::to_comma_seperated;

int main() {
  Device dev;
  Options options = Options(
      {options::GpuArchitecture(device.properties()), options::FMAD(false)});
  Program program(load("./examples/program.cu"));

  std::vector<ProgramArg> program_args{};
  int[] array{5, 3, 3, 2, 7};
  int data {8};
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

  std::vector<ValueType> vec(array, array + 5);
  std::cout << to_comma_seperated(vec) << std::endl;

  return 0;
}
