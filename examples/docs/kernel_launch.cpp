#include "cudaexecutor/main.hpp"

using cudaexecutor::Source, cudaexecutor::KernelArg, cudaexecutor::Kernel,
    cudaexecutor::Device;

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
args.emplace_back(KernelArg{&N});

dim3 grid(NUM_BLOCKS);
dim3 block(NUM_THREADS);

Program program = source.program("saxpy");
Kernel kernel = program.compile();
kernel.configure(grid, block);
kernel.launch(args);

// Alternatively method chaining

source.program("saxpy")
      .compile()
      .configure(grid, block)
      .launch(args);