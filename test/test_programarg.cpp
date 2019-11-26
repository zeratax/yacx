#include "../include/cudaexecutor/ProgramArg.hpp"

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

using cudaexecutor::ProgramArg;

TEST_CASE("ProgramArg can be constructed", "[cudaexecutor::ProgramArg]") {
  float a{5.1};
  float *hX = new float[5]{1, 2, 3, 4, 5};
  float *hY = new float[5]{6, 7, 8, 9, 10};
  float *hOut = new float[5]{11, 12, 13, 14, 15};
  size_t bufferSize = 5 * sizeof(float);

  std::vector<ProgramArg> program_args;
  program_args.emplace_back(ProgramArg(&a));
  // program_args.emplace_back(ProgramArg(hX.data(), bufferSize));
  // program_args.emplace_back(ProgramArg(hY.data(), bufferSize));
  // program_args.emplace_back(ProgramArg(hOut.data(), bufferSize));
  program_args.emplace_back(ProgramArg{&hX, bufferSize});
  program_args.emplace_back(ProgramArg{&hY, bufferSize});
  program_args.emplace_back(ProgramArg{&hOut, bufferSize, true});

  CUdevice cuDevice;
  CUcontext context;
  CUmodule module;
  CUfunction kernel;
  CUDA_SAFE_CALL(cuInit(0));
  CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));
  CUDA_SAFE_CALL(cuCtxCreate(&context, 0, cuDevice));
  //  (cuModuleLoadDataEx(&module, ptx, 0, 0, 0));
  //  (cuModuleGetFunction(&kernel, module, "saxpy"));

  SECTION("ProgramArg can be uploaded and downloaded") {
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
