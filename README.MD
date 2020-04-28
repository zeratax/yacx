[![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/ZerataX/yacx?include_prereleases&style=for-the-badge)](https://github.com/ZerataX/yacx/releases)
 [![tests](https://img.shields.io/github/workflow/status/ZerataX/yacx/tests?label=tests&style=for-the-badge)](https://github.com/ZerataX/yacx/actions?query=workflow%3A%22tests%22) [![Matrix](https://img.shields.io/matrix/yacx:matrix.org?label=%23yacx%3Admnd.sh&style=for-the-badge)](https://matrix.to/#/#yacx:dmnd.sh)
# yacx - Yet Another CudaExecutor

wrapper to easily compile and execute cuda kernels

**features:**
- simple safe syntax
- JNI (with java and scala examples)
- online compilation with NVRTC
- template kernels
- logging system
- descriptive exceptions
- benchmarking
  - time kernel execution as well as uploading and downloading arguments to and from the device

for more see the [changelog](./CHANGELOG.md)

## Requirements

  - [Cuda Toolkit >= V7.0 (recommended 9.3)](https://developer.nvidia.com/cuda-toolkit)
  - C++17 (just to build)
  - JDK >= 8 (to build the JNI)

## C++

### Usage

```c++
// file.cpp
#include "yacx/main.hpp"
#include <algorithm>
#include <experimental/iterator>
#include <vector>

using yacx::Source, yacx::KernelArg, yacx::Options, yacx::Device, yacx::Devices, yacx::type_of;

int main() {
  const int data{1};
  const int times{4};
  const size_t size{32};
  
  static_assert(!(size % times));
  
  std::vector<int> v;
  v.resize(size);
  std::fill(v.begin(), v.end(), 0);
  
  try {
    Device device = Devices::findDevice();
    Options options{yacx::options::GpuArchitecture(device),
                    yacx::options::FMAD(false)};
    options.insert("--std", "c++14");
    Source source{
        "template<typename type, int size>\n"
        "__global__ void my_kernel(type* c, type val) {\n"
        "    auto idx{blockIdx.x * blockDim.x + threadIdx.x};\n"
        "\n"
        "    #pragma unroll(size)\n"
        "    for (auto i{0}; i < size; ++i) {\n"
        "        c[i] = idx + val;\n"
        "    }\n"
        "}"};

    std::vector<KernelArg> args;
    args.emplace_back(
        KernelArg{v.data(), sizeof(int) * v.size(), true});
    args.emplace_back(KernelArg{data});

    dim3 grid(v.size()/times);
    dim3 block(1);
    source.program("my_kernel")
        .instantiate(type_of(data), times)
        .compile(options)
        .configure(grid, block)
        .launch(args, device);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    exit(1);
  }

  std::cout << '\n';
  std::copy(v.begin(), v.end(),
            std::experimental::make_ostream_joiner(std::cout, ", "));
  std::cout << std::endl;
   
  // 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8

  return 0;
}
```

#### Compile

download the latest version of the library from the [releases](https://github.com/ZerataX/yacx/releases/latest) section or build it yourself:

```console
cmake -H. -Bbuild
make -C build yacx
# additionally for the JNI
make -C build yacx-jni
```

and then compile with:
```console 
g++ -lyacx -lnvrtc -lcuda -L $(CUDA_PATH)/lib64 -Wl,-rpath,$(CUDA_PATH)/lib64 file.cpp
```

some syntatic sugar is given using templates, which requires you to include the headers when building:
```console 
g++ -Wall -Wextra --pedantic -lyacx -lnvrtc -lcuda -L $(CUDA_PATH)/lib64 -Wl,-rpath,$(CUDA_PATH)/lib64 -I include/yacx file.cpp
```

### Run Examples

```console
# list examples
$ find examples -maxdepth 1 -name "*.cpp"
examples/example_template.cpp
examples/example_matrix_multiply.cpp
examples/example_saxpy.cpp
examples/example_program.cpp
examples/example_gauss.cpp
# compile example
$ cmake -H. -Bbuild
$ make -C build example_program
# run example
$ ./build/examples/bin/example_program
1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8
```

### Tests

Require the [Catch2](https://github.com/catchorg/Catch2) test-framework

```console
$ cmake -H. -Bbuild
$ make -C build tests
$ pushd build/test/bin/
$ ./tests
$ popd
```

For more info read the [contribution guideline](https://github.com/ZerataX/yacx/blob/master/CONTRIBUTING.md#test)

## Java

### Run Examples

```console
# for autocompletion
$ source yacx-completion.bash
# compile example
$ ./yacx build-java
# list examples
$ ./yacx execute-java
!! parameter needed, select one of the following
ExampleSimpleGEMMBenchmark
ExampleFastGEMM
ExampleSimpleGEMM
...
$ ./yacx.sh execute-java ExampleFastGEMM
```

### Tests

```console
# build JNI and Test Classes
$ cmake -H. -Bbuild
$ make -C build JNITestClasses
# link kernels to execution folder
$ mkdir build/java/bin/examples
$ ln -s $(realpath examples/kernels/) build/java/bin/examples/kernels
# execute tests
$ pushd build/java/bin
$ java -Djava.library.path=../../ -jar src/junit-platform-console-standalone-1.5.2.jar --class-path . --scan-class-path
$ popd
```

## Scala

### Run Examples

```console
# for autocompletion
$ source yacx-completion.bash
# compile example
$ ./yacx build-scala
# list examples
$ ./yacx execute-scala
!! parameter needed, select one of the following
ExampleFilterBenchmark
ExampleFilter
ExampleFastGEMM
...
$ ./yacx.sh execute-scala ExampleFastGEMM
```

## Contributing

Make sure you agree to the [code of conduct](./CODE_OF_CONDUCT.md) and read the [contribution guideline](https://github.com/ZerataX/yacx/blob/master/CONTRIBUTING.md)

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
