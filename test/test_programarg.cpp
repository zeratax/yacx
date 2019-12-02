#include "cudaexecutor/KernelArg.hpp"

#include <catch2/catch.hpp>
#include <cuda.h>
#include <iostream>

#define CUDA_SAFE_CALL(error)                                                  \
  do {                                                                         \
    CUresult result = error;                                                   \
    if (result != CUDA_SUCCESS) {                                              \
      const char *msg;                                                         \
      cuGetErrorName(result, &msg);                                            \
      std::cerr << "\nerror: " #error " failed with error " << msg << '\n';    \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

using cudaexecutor::KernelArg;

TEST_CASE("KernelArg can be constructed", "[cudaexecutor::KernelArg]") {
  float a{5.1};
  float *hX = new float[5]{1, 2, 3, 4, 5};
  float *hY = new float[5]{6, 7, 8, 9, 10};
  float *hOut = new float[5]{11, 12, 13, 14, 15};
  size_t bufferSize = 5 * sizeof(float);

  std::vector<KernelArg> program_args;
  program_args.emplace_back(KernelArg(&a));
  // program_args.emplace_back(KernelArg(hX.data(), bufferSize));
  // program_args.emplace_back(KernelArg(hY.data(), bufferSize));
  // program_args.emplace_back(KernelArg(hOut.data(), bufferSize));
  program_args.emplace_back(KernelArg{&hX, bufferSize});
  program_args.emplace_back(KernelArg{&hY, bufferSize});
  program_args.emplace_back(KernelArg{&hOut, bufferSize, true});

  CUdevice cuDevice;
  CUcontext context;
  CUmodule module;
  CUfunction kernel;
  CUDA_SAFE_CALL(cuInit(0));
  CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));
  CUDA_SAFE_CALL(cuCtxCreate(&context, 0, cuDevice));
  //  (cuModuleLoadDataEx(&module, ptx, 0, 0, 0));
  //  (cuModuleGetFunction(&kernel, module, "saxpy"));

  SECTION("KernelArg can be uploaded and downloaded") {
    for (auto &arg : program_args)
      arg.upload();

    hX[0] = 0;
    hY[0] = 0;
    hOut[0] = 0;

    for (auto &arg : program_args)
      arg.download();

    REQUIRE(hX[0] == 0);
    REQUIRE(hY[0] == 0);
    REQUIRE(hOut[0] == 11);
  }

  CUDA_SAFE_CALL(cuCtxDestroy(context));

  delete[] hX;
  delete[] hY;
  delete[] hOut;
}
