#include "yacx/main.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <ctime>
#include <functional>
#include <iostream>
#include <iterator>
#include <random>
#include <string>

using yacx::Headers, yacx::Header, yacx::Source, yacx::KernelArg, yacx::Kernel,
    yacx::Options, yacx::Device, yacx::type_of, yacx::KernelTime, yacx::Devices,
    yacx::arg_type, yacx::load;

// A function template to compare the values of two array elements.
template <typename type> void compare(type *lhs, type *rhs, int array_size) {
  // 1. Declaring the variables prior to comparing 2 array elements
  int errors{0};
  float lhs_type, rhs_type;
  std::string errors_string, array_size_string, i_string, lhs_string,
      rhs_string;

  /*2A. Calculate the number of errors caused by inconsistencies between
        values of two array elements.*/
  for (int i{0}; i < array_size; i += 1) {
    lhs_type = static_cast<int>(lhs[i]);
    rhs_type = static_cast<int>(rhs[i]);

    if ((lhs_type - rhs_type) != 0) {
      errors += 1;
      std::cout << i << " expected " << lhs[i] << ": actually " << rhs[i]
                << std::endl;
    }
  }

  /*2B. Determining, whether there are errors in the consistencies of values
        of two array elements.*/
  errors_string = static_cast<int>(errors);
  array_size_string = static_cast<int>(array_size);

  if (errors > 0) {
    std::cout << yacx::gColorBrightRed << errors_string
              << " errors occured,"
                 " out of "
              << array_size_string << " values." << yacx::gColorReset
              << std::endl;
  } else {
    std::cout << yacx::gColorBrightGreen << "No errors occured."
              << yacx::gColorReset << std::endl;
  }
}

// Template function for comparing the values of two array elements
template <typename type>
std::function<bool(const type &, const type &)> comparator =
    [](const type &left, const type &right) {
      double epsilon{1.0E-8};
      float lhs_type = static_cast<float>(left),
            rhs_type = static_cast<float>(right);
      return (abs(lhs_type - rhs_type) < epsilon);
    };

// A template function to fill all array elements with random values
template <typename type>
void fillArray(type *array, int array_size, int min = 0, int max = 100) {
  // 1. Declaring and initialising variables required to run the template
  // function
  static std::random_device rd;
  static std::mt19937 mte(rd());
  std::uniform_int_distribution<int> dist(min, max);

  // 2. Generating random values using random value generator
  for (int i = 0; i < array_size; ++i)
    array[i] = static_cast<type>(dist(mte));
}

// Template function to calculate 2 arrays of similar length on the host-side.
template <typename type>
void sumArrayOnHost(const type *A, const type *B, type *C, const size_t nx,
                    const size_t ny, const size_t nz) {
  // 1. Declaring the variables required to add values from two 3D-array
  // elements
  size_t i;

  // 2. Loop to calculate the sum of values of two array elements
  for (size_t ix = 0; ix < nx; ++ix) {
    for (size_t iy = 0; iy < ny; ++iy) {
      for (size_t iz = 0; iz < nz; ++iz) {
        i = iz * (nz * ny) + iy * ny + ix;
        C[i] = A[i] + B[i];
      }
    }
  }
}

// A general example of the functionalities of a CUDA-Executor
int main() {
  /*1. Declaring and initialising the variables required to run the
       CUDA-Executor-example*/
  std::clock_t start;
  bool equalSumArray;
  const float data{1.69f};
  const size_t NX{800}, NY{200}, NZ{200};
  const size_t BLOCK_SIZE{8};
  const size_t GRID_SIZE{4};
  const size_t matrix_size{(NX * NY * NZ) * sizeof(float)};

  /*1A. Checking the correctness of the modulus of NX,NY,NZ with
        BLOCK_SIZE*/
  static_assert(NX % BLOCK_SIZE == 0);
  static_assert(NY % BLOCK_SIZE == 0);
  static_assert(NZ % BLOCK_SIZE == 0);
  static_assert(NY * GRID_SIZE == NX);
  static_assert(NZ * GRID_SIZE == NX);

  float *A = new float[NX * NY * NZ];
  float *B = new float[NX * NY * NZ];
  float *C_seq = new float[NX * NY * NZ];
  float *C_cuda = new float[NX * NY * NZ];

  KernelTime time;

  // 1B. Fill arrays with random values
  fillArray<float>(A, NX * NY * NZ);
  fillArray<float>(B, NX * NY * NZ);

  // 2. Running the CUDA-Executor-example
  Device dev = Devices::findDevice();
  try {
    // 2A. Select Device
    std::cout << "===================================\n";
    std::cout << "Selected " << dev.name() << " with "
              << (dev.total_memory() / 1024) / 1024 << "mb VRAM\n";
    std::cout << "Kernel Arguments total size: "
              << ((matrix_size * 3 + sizeof(float)) / 1024 / 1024) << "mb\n" << std::endl;

    // 2B. Set kernel string, header files and compile options
    Headers headers;
    headers.insert(Header{"cuda_runtime.h"});
    Source source{load("kernels/sum_Array.cu"), headers};

    Options options;
    options.insert("--std", "c++14");
    options.insert("--device-debug");
    options.insertOptions(yacx::options::GpuArchitecture{dev});
    options.insertOptions(yacx::options::FMAD(true));

    // 2C. Set kernel arguments
    std::vector<KernelArg> args;
    args.emplace_back(KernelArg{A, matrix_size});
    args.emplace_back(KernelArg{B, matrix_size});
    args.emplace_back(KernelArg{C_cuda, matrix_size, true, false});

    // 2D. Compile Kernels
    dim3 block(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(GRID_SIZE);

    Kernel kernel_sumArray = source.program("sumArrayOnGPU")
                                 .instantiate(type_of(data))
                                 .compile(options)
                                 .configure(grid, block);

    // 2E. Launch kernels

    // CPU single threaded sum array
    start = std::clock();
    sumArrayOnHost<float>(A, B, C_seq, NX, NY, NZ);
    std::cout << "Time" << yacx::gColorBrightYellow << "[CPU single threaded]"
              << yacx::gColorReset << ":   "
              << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000)
              << " ms" << std::endl;

    time = kernel_sumArray.launch(args);

    equalSumArray =
        std::equal(C_cuda, C_cuda + (NX + NY + NY), C_seq, comparator<float>);

    if (!equalSumArray)
      compare<float>(C_seq, C_cuda, NX + NY + NZ);

  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    exit(1);
  }

  std::cout << "Theoretical Bandwith:        "
            << yacx::theoretical_bandwidth(dev) << " GB/s\n";
  std::cout << "Effective Bandwith:          "
            << time.effective_bandwidth_launch() << " GB/s\n";
  std::cout << "Time" << yacx::gColorBrightYellow << "[GPU multi threaded]"
            << yacx::gColorReset << ":\n";
  std::cout << yacx::gColorGray << time << yacx::gColorReset << std::endl;
  if (equalSumArray) {
    std::cout << yacx::gColorBrightGreen
              << "Everything was correctly calculated!" << yacx::gColorReset
              << std::endl;
  } else {
    std::cout << yacx::gColorBrightRed << "SumArray went wrong ;_;"
              << yacx::gColorReset << std::endl;
  }

  std::cout << yacx::gColorReset
            << "===================================" << std::endl;

  // Free resources
  delete[] A;
  delete[] B;
  delete[] C_seq;
  delete[] C_cuda;

  return 0;
}