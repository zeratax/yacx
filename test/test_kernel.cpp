// Here, the sources of kernels are tested.
#include "yacx/Exception.hpp"
#include "yacx/Headers.hpp"
#include "yacx/Kernel.hpp"
#include "yacx/KernelArgs.hpp"
#include "yacx/Source.hpp"

#include <catch2/catch.hpp>
#include <exception>
#include <iostream>
#include <signal.h>

using yacx::Kernel, yacx::KernelTime, yacx::KernelArg, yacx::Source,
    yacx::Header, yacx::Headers;


TEST_CASE(
    "The kernel - source code will be tested under the following conditions.") {
  // A. Preparing the input for the kernel-compilation using source
  int datasize{10};
  int *hX = new int[10]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  int *hY = new int[10]{6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  int *hOut_1 = new int[10]{7, 9, 11, 13, 15, 1, 2, 3, 4, 5};
  int *hOut_2 = new int[10]{7, 9, 11, 13, 15, 1, 2, 3, 4, 5};
  int *hOut_3 = new int[10]{7, 9, 11, 13, 15, 1, 2, 3, 4, 5};
  int *hOut_4 = new int[10]{7, 9, 11, 13, 15, 1, 2, 3, 4, 5};

  size_t bufferSize = datasize * sizeof(int);

  std::vector<KernelArg> args;
  args.emplace_back(KernelArg{hX, bufferSize});
  args.emplace_back(KernelArg{hY, bufferSize});
  args.emplace_back(KernelArg{hOut_1, bufferSize, true});
  args.emplace_back(KernelArg(&datasize));

  Headers headers;
  headers.insert(Header{"cuda_runtime.h"});

  // A2. Preparing the output for kernel-compilation
  int *hostCompareOutput = new int[10]{7, 9, 11, 13, 15, 17, 19, 21, 23, 25};

  // B1. A Kernel must be created in a controlled manner.
  SECTION("1. A Kernel must be created in a controlled manner.") {
    Source source{"#include \"cuda_runtime.h\"\n"
                  "extern \"C\"\n"
                  "__global__ void cuda_add(int *x, int *y, int *out, int "
                  "datasize) {\n"
                  " int i = threadIdx.x;\n"
                  " out[i] = x[i] + y[i];\n"
                  "}",
                  headers};

    dim3 grid_test(1);
    dim3 block_test(10);

    // B1A1. Controlled results from controlled inputs.
    REQUIRE_NOTHROW(source.program("cuda_add")
                        .compile()
                        .configure(grid_test, block_test)
                        .launch(args));

    for (int i = 0; i < 10; i++) {
      REQUIRE(hOut_1[i] == hostCompareOutput[i]);
    }
  }

  // B2A. Checking for consistencies in the specification of block and grid
  // dimensions.
  SECTION("2B. The created kernel is launched using proper block and grid "
          "dimensions.") {
    Source source{"#include \"cuda_runtime.h\"\n"
                  "extern \"C\"\n"
                  "__global__ void cuda_add(int *x, int *y, int *out, int "
                  "datasize) {\n"
                  " int i = threadIdx.x;\n"
                  " out[i] = x[i] + y[i];\n"
                  "}",
                  headers};

    dim3 grid_test(1);
    dim3 grid_test_negative(-2);
    dim3 grid_test_zero(0);

    dim3 block_test(10);
    dim3 block_test_negative(-2);
    dim3 block_test_zero(0);

    // B2B1. Launching created kernels with improper grid and block
    // configurations.
    REQUIRE_THROWS(source.program("cuda_add")
                       .compile()
                       .configure(grid_test_negative, block_test_negative)
                       .launch(args));

    REQUIRE_THROWS(source.program("cuda_add")
                       .compile()
                       .configure(grid_test_zero, block_test_zero)
                       .launch(args));

    // B2B2. Launching created kernels with proper grid and block
    // configurations.
    REQUIRE_NOTHROW(source.program("cuda_add")
                        .compile()
                        .configure(grid_test, block_test)
                        .launch(args));

    KernelTime kernel_launching = source.program("cuda_add")
                                      .compile()
                                      .configure(grid_test, block_test)
                                      .launch(args);

    // B2B3. Comparing the results
    // B2B4. Checking for consistencies between given output constraints and
    // host results
    for (int i = 0; i < 10; i++) {
      REQUIRE(hOut_1[i] == hostCompareOutput[i]);
    }

    // B2B5. Ensuring that the kernel is launched
    REQUIRE(kernel_launching.sum > 0);
    REQUIRE(kernel_launching.download > 0);
    REQUIRE(kernel_launching.launch > 0);
  }

  // B2C. Checking for consistencies in the specification of block and grid
  // dimensions.
  SECTION("2C. The created kernel is launched using different block and grid "
          "dimensions.") {
    // B2C1. Declaration of all necessary kernel function inputs.
    Source source{"#include \"cuda_runtime.h\"\n"
                  "extern \"C\"\n"
                  "__global__ void cuda_add(int *x, int *y, int *out, int "
                  "datasize) {\n"
                  " int i = threadIdx.x;\n"
                  " out[i] = x[i] + y[i];\n"
                  "}",
                  headers};

    int counter = 1;

    dim3 grid_block_no1(1), grid_block_no2(2), grid_block_no3(5);

    source.program("cuda_add")
        .compile()
        .configure(grid_block_no1, grid_block_no3)
        .launch(args);

    // B2C2. Testing kernel - launch with various block and grid dimensions.
    // Resetting.
    args.clear();
    args.emplace_back(KernelArg{hX, bufferSize});
    args.emplace_back(KernelArg{hY, bufferSize});
    args.emplace_back(KernelArg{hOut_2, bufferSize, true});
    args.emplace_back(KernelArg(&datasize));

    source.program("cuda_add")
        .compile()
        .configure(grid_block_no3, grid_block_no1)
        .launch(args);

    // Resetting.
    args.clear();
    args.emplace_back(KernelArg{hX, bufferSize});
    args.emplace_back(KernelArg{hY, bufferSize});
    args.emplace_back(KernelArg{hOut_3, bufferSize, true});
    args.emplace_back(KernelArg(&datasize));

    source.program("cuda_add")
        .compile()
        .configure(grid_block_no3, grid_block_no2)
        .launch(args);

    // Resetting.
    args.clear();
    args.emplace_back(KernelArg{hX, bufferSize});
    args.emplace_back(KernelArg{hY, bufferSize});
    args.emplace_back(KernelArg{hOut_4, bufferSize, true});
    args.emplace_back(KernelArg(&datasize));

    source.program("cuda_add")
        .compile()
        .configure(grid_block_no2, grid_block_no3)
        .launch(args);

    // B2C3. Comparing the results
    // A. With hOut1
    for (int i = 0; i < 5; i++)
      REQUIRE(hOut_1[i] == hostCompareOutput[i]);

    for (int j = 5; j < 10; j++) {
      REQUIRE(hOut_1[j] == counter);
      counter++;
    }

    // B. With hOut2
    counter = 1;

    for (int i = 0; i < 5; i++)
      REQUIRE(hOut_2[i] == hostCompareOutput[i]);

    for (int j = 5; j < 10; j++) {
      REQUIRE(hOut_2[j] == counter);
      counter++;
    }

    // C. With hOut3
    counter = 1;

    for (int i = 0; i < 5; i++)
      REQUIRE(hOut_3[i] == hostCompareOutput[i]);

    for (int j = 5; j < 10; j++) {
      REQUIRE(hOut_3[j] == counter);
      counter++;
    }

    // D. With hOut4
    counter = 1;

    for (int i = 0; i < 5; i++)
      REQUIRE(hOut_4[i] == hostCompareOutput[i]);

    for (int j = 5; j < 10; j++) {
      REQUIRE(hOut_4[j] == counter);
      counter++;
    }
  }

  // Free all storage memory
  delete[] hX;
  delete[] hY;
  delete[] hOut_1;
  delete[] hOut_2;
  delete[] hOut_3;
  delete[] hOut_4;
  delete[] hostCompareOutput;
}
