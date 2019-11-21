# CUDA EXECUTOR

wrapper to easily compile and execute cuda kernels

## Usage

```c++
#include <array>
#include "../include/cudaexecutor/main.hpp"

using cudaexecutor::Program, cudaexecutor::ProgramArg, cudaexecutor::Kernel,
    cudaexecutor::Options, cudaexecutor::Device, cudaexecutor::load,
    cudaexecutor::type_of, cudaexecutor::to_comma_separated;

int main() {
  std::array<int, 5> array{5, 3, 3, 2, 7};
  int data{8};
  std::string kernel_string = "template<typename type, int size>"\
    "__global__ void setKernel(type[] c, type val) {"\
    "   auto idx = threadIdx.x * size;"\
    "   #pragma unroll(size)"\
    "   for (auto i = 0; i < size; i++) {"\
    "       c[idx] = val;"\
    "       idx++;"\
    "   }"\
    "}"

  try {
    Device device;
    Options options{cudaexecutor::options::GpuArchitecture(device),
                    cudaexecutor::options::FMAD(false)};
    // Program program{load("./examples/kernel/program.cu")};
    Program program{kernel_string};

    std::vector<ProgramArg> program_args;
    program_args.emplace_back(ProgramArg{array.data(), sizeof(int) * 5, true});
    program_args.emplace_back(ProgramArg{&data});

    dim3 grid(1);
    dim3 block(1);
    program
        .kernel("my_kernel")
        .instantiate(type_of(data), 5)
        .compile(options)
        .configure(grid, block)
        .launch(program_args);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
  }

  std::vector<int> vec(array.begin(), array.end());
  std::cout << to_comma_separated(vec) << std::endl;

  return 0;
}
```

### Run Examples

```console
$ make example1
$ ./bin/runner
```

### Tests
Makes use of the [Catch2](https://github.com/catchorg/Catch2) test-framework

```console
$ make check
```

For more info run `./bin/tester -?`

## Resources

- OpenCL to CUDA:
  - <https://www.sharcnet.ca/help/index.php/Porting_CUDA_to_OpenCL>
- DriverAPI & NVRTC:
  - <https://docs.nvidia.com/cuda/cuda-driver-api/driver-vs-runtime-api.html#driver-vs-runtime-api>
  - <https://docs.nvidia.com/cuda/nvrtc/index.html>
- Similar Projects:
  - <https://github.com/mgopshtein/cudacpp/blob/master/examples/invoke-rtc.cpp>
  - <https://github.com/NVIDIA/jitify>
