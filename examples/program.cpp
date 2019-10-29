#include "Program.hpp"
#include "Exception.hpp"
#include "Kernel.hpp"

#include <cuda.h>

using cudaexecutor::Program, cudaexecutor::ProgramArg, cudaexecutor::Kernel,
    cudaexecutor::load;

int main() {
  Kernel kernel;
  CudaDevice dev = CudaDevice;
  Options options = Options(
      {options::GpuArchitecture(device.properties()), options::FMAD(false)});

  // compile erst nach instantiate
  Program program = kernel.compile(load("./examples/program.cu"), options);
  std::vector<ProgramArg> program_args{};

  int[] array{5, 3, 3, 2, 7};
  int data{5};
  ProgramArg array_arg(&array, true);
  ProgramArg data_arg(&data);
  program_args.push_back(array_arg);
  program_args.push_back(data_arg);

  dim3 grid(1);
  dim3 block(1);
  try {
    program.kernel("my_kernel");
    .instantiate(type_of(data), 5);
    .compile(options);
    .configure(grid, block);
    .launch(program_args);
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
  }

  for (auto element : &array)
    std::cout << element << std::endl;

  return 0;
}
