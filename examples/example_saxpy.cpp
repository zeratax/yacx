#include "../include/cudaexecutor/main.hpp"

#include <memory>

using cudaexecutor::Program, cudaexecutor::ProgramArg, cudaexecutor::Kernel,
    cudaexecutor::Options, cudaexecutor::Device, cudaexecutor::load,
    cudaexecutor::type_of, cudaexecutor::to_comma_separated;

int main() {
  size_t SIZE = 5;
  float a{8};
  std::array<float, 5> x{5, 3, 3, 2, 7};
  std::array<float, 5> y{9, 4, 2, 5, 1};
  std::array<float, 5> out{};
  try {
    Program program{"extern \"C\" __global__\n"
        "void saxpy(float a, float *x, float *y, float *out, size_t n) {\n"
        "  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "  if (tid < n) {\n"
        "    out[tid] = a * x[tid] + y[tid];\n"
        "  }\n"
        "}"};

    std::vector<ProgramArg> program_args;
    program_args.emplace_back(&a, sizeof(float));
    program_args.emplace_back(x.data(), sizeof(float) * SIZE);
    program_args.emplace_back(y.data(), sizeof(float) * SIZE);
    program_args.emplace_back(ProgramArg(out.data(), sizeof(float) * SIZE, true));
    program_args.emplace_back(ProgramArg(&SIZE, sizeof(size_t)));


    dim3 grid(1);
    dim3 block(1);
    program
        .kernel("saxpy")
        .compile()
        .configure(grid, block)
        .launch(program_args);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
  }

  std::vector<int> vec(out.data(), out.data() + 5);
  std::cout << to_comma_separated(vec) << std::endl;

  return 0;
}
