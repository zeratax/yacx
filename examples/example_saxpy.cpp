#include "../include/cudaexecutor/main.hpp"

#include <memory>

#define NUM_THREADS 256
#define NUM_BLOCKS 32

using cudaexecutor::Program, cudaexecutor::ProgramArg, cudaexecutor::Kernel,
    cudaexecutor::Options, cudaexecutor::Device, cudaexecutor::load,
    cudaexecutor::type_of, cudaexecutor::to_comma_separated;

int main() {
  const float DELTA = 0.01f;
  size_t SIZE = 5;
  size_t n = NUM_THREADS * NUM_BLOCKS;
  size_t bufferSize = n * sizeof(float);
  float a = 5.1f;
  std::array<float, NUM_THREADS * NUM_BLOCKS> hX, hY, hOut;
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
    program_args.emplace_back(ProgramArg{&a});
    program_args.emplace_back(ProgramArg(hX.data(), bufferSize));
    program_args.emplace_back(ProgramArg(hY.data(), bufferSize));
    program_args.emplace_back(ProgramArg(hOut.data(), bufferSize, true, false));
    program_args.emplace_back(ProgramArg{&n});

    dim3 grid(NUM_BLOCKS);
    dim3 block(NUM_THREADS);
    Kernel kernel = program.kernel("saxpy");
        kernel.compile();
        kernel.configure(grid, block);
        kernel.launch(program_args);
  } catch (const std::exception &e) {
    std::cerr << "exception caught: " << e.what() << std::endl;
  }

  bool correct = true;
  for (int j = 0; j < hOut.size(); ++j) {
    float expected = hX.at(j) * a + hY.at(j);
    if ((expected - hOut.at(j)) > DELTA) {
      correct = false;
      std::cout << "Expected: " << expected << " != "
                << " Result: " << hOut.at(j) << std::endl;
    }   
  }

  if (correct)
    std::cout << "Everything was calculated correctly!!!" << std::endl;

  return 0;
}
