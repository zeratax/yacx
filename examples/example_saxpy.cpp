#include "../include/cudaexecutor/main.hpp"

#include <memory>

#define NUM_THREADS 16
#define NUM_BLOCKS 32

using cudaexecutor::Program, cudaexecutor::ProgramArg, cudaexecutor::Kernel,
    cudaexecutor::Options, cudaexecutor::Device, cudaexecutor::load,
    cudaexecutor::type_of, cudaexecutor::to_comma_separated;

int main() {
  size_t SIZE = 5;
  size_t n = NUM_THREADS * NUM_BLOCKS;
  size_t bufferSize = n * sizeof(float);
  float a = 5.1f;
  float *hX = new float[n], *hY = new float[n], *hOut = new float[n];
  //  std::array<float, NUM_THREADS * NUM_BLOCKS> hX, hY, hOut;
  for (size_t i = 0; i < n; ++i) {
    hX[i] = static_cast<float>(i);
    hY[i] = static_cast<float>(i * 2);
  }


  try {
    Program program{
        "extern \"C\" __global__\n"
        "void saxpy(float a, float *x, float *y, float *out, size_t n) {\n"
        "  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "  if (tid < n) {\n"
        "    out[tid] = a * x[tid] + y[tid];\n"
        "  }\n"
        "}"};

    std::vector<ProgramArg> program_args;
    program_args.emplace_back(ProgramArg(&a));
    program_args.emplace_back(ProgramArg(&hX, bufferSize, false, true));
    program_args.emplace_back(ProgramArg{&hY, bufferSize, false, true});
    program_args.emplace_back(ProgramArg{&hOut, bufferSize, true, false});
    program_args.emplace_back(ProgramArg(&n));

    dim3 grid(NUM_BLOCKS);
    dim3 block(NUM_THREADS);
    program.kernel("saxpy")
        .compile()
        .configure(grid, block)
        .launch(program_args);
  } catch (const std::exception &e) {
      std::cerr << "Error:" << std::endl;
      std::cerr << e.what() << std::endl;
  }

  for (int j = 0; j < n; ++j) {
    float expected = hX[j] * a + hY[j];
    if (expected != hOut[j])
      std::cout << "Expected: " << expected << " != "
                << " Result: " << hOut[j] << std::endl;
  }

  // std::vector<int> vec(hOut.data(), hOut.data() + n);
  // std::cout << "Result: " << to_comma_separated(vec) << std::endl;

  delete[] hX;
  delete[] hY;
  delete[] hOut;

  return 0;
}
