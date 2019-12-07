# yacx - Yet Another CudaExecutor

wrapper to easily compile and execute cuda kernels

## Usage

```c++
#include "yacx/main.hpp"
#include <array>
#include <experimental/iterator>

using yacx::Source, yacx::KernelArg, yacx::Kernel,
    yacx::Options, yacx::Device, yacx::load,
    yacx::type_of;

int main() {
  std::array<int, 32> array;
  array.fill(0);
  int data{1};
  try {
    Device device;
    Options options{yacx::options::GpuArchitecture(device),
                    yacx::options::FMAD(false)};
    options.insert("--std", "c++14");
    Source source{
        "template<typename type, int size>\n"
        "__global__ void my_kernel(type* c, type val) {\n"
        "    auto idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "\n"
        "    #pragma unroll(size)\n"
        "    for (auto i = idx * size; i < idx * size + size; i++) {\n"
        "        c[i] = idx + val;\n"
        "    }\n"
        "}"};

    std::vector<KernelArg> args;
    args.emplace_back(
        KernelArg{array.data(), sizeof(int) * array.size(), true});
    args.emplace_back(KernelArg{&data});

    dim3 grid(8);
    dim3 block(1);
    source.program("my_kernel")
        .instantiate(type_of(data), 4)
        .compile(options)
        .configure(grid, block)
        .launch(args, device);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    exit(1);
  }

  std::cout << '\n';
  std::copy(array.begin(), array.end(),
            std::experimental::make_ostream_joiner(std::cout, ", "));
  std::cout << std::endl;
   
  // 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8

  return 0;
}
```

### Run Examples

```console
$ make example_program
$ make run
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
  - [Jitify](https://github.com/NVIDIA/jitify)
  - [OCAL](https://doi.org/10.1109/PADSW.2018.8644541)
