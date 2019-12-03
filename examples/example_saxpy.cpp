#include "../include/cudaexecutor/main.hpp"
#include "cudaexecutor/main.hpp"

#define NUM_THREADS 512
#define NUM_BLOCKS 1024

using cudaexecutor::Source, cudaexecutor::KernelArg, cudaexecutor::Kernel,
    cudaexecutor::KernelTime, cudaexecutor::Device, cudaexecutor::load;

int main() {
  const float DELTA{0.01f};
  const size_t N{NUM_THREADS * NUM_BLOCKS};
  size_t bufferSize{N * sizeof(float)};
  float a{5.1f};
  std::array<float, N> hX, hY, hOut;
  for (size_t i{0}; i < N; ++i) {
    hX.at(i) = static_cast<float>(i * 0.01);
    hY.at(i) = static_cast<float>(i * 0.02);
  }

  KernelTime time;

  try {
    Device dev;
    Source source{
        "extern \"C\" __global__\n"
        "void saxpy(float a, float *x, float *y, float *out, size_t n) {\n"
        "  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "  if (tid < n) {\n"
        "    out[tid] = a * x[tid] + y[tid];\n"
        "  }\n"
        "}"};

    std::vector<KernelArg> args;
    args.emplace_back(KernelArg{&a});
    args.emplace_back(KernelArg{hX.data(), bufferSize});
    args.emplace_back(KernelArg{hY.data(), bufferSize});
    args.emplace_back(KernelArg{hOut.data(), bufferSize, true, false});
    args.emplace_back(KernelArg{const_cast<size_t *>(&N)});

    std::cout << "Selected " << dev.name() << " with "
              << (dev.total_memory() / 1024) / 1024 << "mb" << std::endl;
    std::cout << "Arguments have a combined size of "
              << ((bufferSize * 3 + 2 * sizeof(int)) / 1024) << "kb"
              << std::endl;

    dim3 grid(NUM_BLOCKS);
    dim3 block(NUM_THREADS);
    time = source.program("saxpy")
               .compile()
               .configure(grid, block)
               .launch(args, dev);
  } catch (const std::exception &e) {
    std::cerr << "Error:\n" << e.what() << std::endl;
    exit(1);
  }

  bool correct = true;

  for (size_t j = 0; j < hOut.size(); ++j) {
    float expected = hX.at(j) * a + hY.at(j);
    if ((expected - hOut.at(j)) > DELTA) {
      correct = false;
      std::cout << "Expected: " << expected << " != "
                << " Result: " << hOut.at(j) << std::endl;
    }
  }

  if (correct)
    std::cout << "Everything was calculated correctly!!!";

  std::cout << "it took " << time.upload "ms to upload all arguments, "
            << time.launch << "ms to execute the kernel and "
            << time.download
               "ms to download the resulting array. Which is a total of"
            << time.sum
            << "ms." << std::endl;

      return 0;
}
