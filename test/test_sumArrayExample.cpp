#include "yacx/Exception.hpp"
#include "yacx/Logger.hpp"
#include "yacx/main.hpp"
#include <catch2/catch.hpp>

using yacx::Kernel, yacx::Source, yacx::KernelArg, yacx::Options, yacx::Device,
    yacx::Devices, yacx::type_of, yacx::loglevel;

TEST_CASE("sumArray with implicit size", "[example_program]") {

  Device device = Devices::findDevice();
  Options options{yacx::options::GpuArchitecture(device),
                  yacx::options::FMAD(false)};
  options.insert("--std", "c++14");
  Source source{
      "extern \"C\" __global__ void sumArrayOnGPU(float *A, float *B, float* "
      "C){\n"
      "int "
      "i_inBlock=threadIdx.x+threadIdx.y*blockDim.x+threadIdx.z*blockDim.y*"
      "blockDim.x;\n"
      "int blockID= blockIdx.x;\n"
      "int i = i_inBlock + blockID*(blockDim.x*blockDim.y*blockDim.z);\n"
      "C[i]=A[i]+B[i];\n"
      "}\n"};

  SECTION("SUM_ARRAY WITH 1 and fixed values") {
    // set up data size of vectors
    int nElem = 1;
    // malloc host memory
    size_t nBytes = nElem * sizeof(float);
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = new float[nBytes];
    h_B = new float[nBytes];
    hostRef = new float[nBytes];
    gpuRef = new float[nBytes];
    // initialize data at host side
    // initialData(h_A, nElem);
    // generate different seed for random number
    for (int i = 0; i < nElem; i++) {
      h_A[i] = 42;
    }
    // initialData(h_B, nElem);
    // generate different seed for random number
    for (int i = 0; i < nElem; i++) {
      h_B[i] = 18;
    }

    std::vector<KernelArg> args;
    args.emplace_back(KernelArg{h_A, nBytes, false});
    args.emplace_back(KernelArg{h_B, nBytes, false});
    args.emplace_back(KernelArg{gpuRef, nBytes, true});

    dim3 block(1, 1, 1);
    dim3 grid(1);
    Kernel k =
        source.program("sumArrayOnGPU").compile(options).configure(grid, block);
    k.launch(args, device);

    // add vector at host side for result checks
    for (int idx = 0; idx < nElem; idx++)
      hostRef[idx] = h_A[idx] + h_B[idx];

    // check device results
    double epsilon = 1.0E-8;
    bool match = 1;
    for (int i = 0; i < nElem; ++i) {
      if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
        match = 0;
        char *error_string;
        asprintf(
            &error_string,
            "Arrays do not match! \n->host %5.2f gpu %5.2f at current %d\n",
            hostRef[i], gpuRef[i], i);
        FAIL(error_string);
        break;
      }
    }
    delete[] h_A;
    delete[] h_B;
    delete[] hostRef;
    delete[] gpuRef;
  }

  // set up data size of vectors
  int nElem = 1024;
  // malloc host memory
  size_t nBytes = nElem * sizeof(float);
  float *h_A, *h_B, *hostRef, *gpuRef;
  h_A = new float[nBytes];
  h_B = new float[nBytes];
  hostRef = new float[nBytes];
  gpuRef = new float[nBytes];

  SECTION("SUM_ARRAY WITH 1024 and fixed values") {
    // initialize data at host side
    // initialData(h_A, nElem);
    // generate different seed for random number
    for (int i = 0; i < nElem; i++) {
      h_A[i] = 42;
    }
    // initialData(h_B, nElem);
    // generate different seed for random number
    for (int i = 0; i < nElem; i++) {
      h_B[i] = 18;
    }

    std::vector<KernelArg> args;
    args.emplace_back(KernelArg{h_A, nBytes, false});
    args.emplace_back(KernelArg{h_B, nBytes, false});
    args.emplace_back(KernelArg{gpuRef, nBytes, true});

    dim3 block(32, 8, 2);
    int blockSize = block.x * block.y * block.z;
    dim3 grid((nElem + blockSize - 1) / blockSize);
    Kernel k =
        source.program("sumArrayOnGPU").compile(options).configure(grid, block);
    k.launch(args, device);

    // add vector at host side for result checks
    for (int idx = 0; idx < nElem; idx++)
      hostRef[idx] = h_A[idx] + h_B[idx];

    // check device results
    double epsilon = 1.0E-8;
    bool match = 1;
    for (int i = 0; i < nElem; ++i) {
      if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
        match = 0;
        char *error_string;
        asprintf(
            &error_string,
            "Arrays do not match! \n->host %5.2f gpu %5.2f at current %d\n",
            hostRef[i], gpuRef[i], i);
        FAIL(error_string);
        break;
      }
    }
  }

  SECTION("SUM_ARRAY WITH 1024 and random values") {
    // initialize data at host side
    // initialData(h_A, nElem);
    // generate different seed for random number
    time_t t;
    srand((unsigned)time(&t));
    for (int i = 0; i < nElem; i++) {
      h_A[i] = (float)(rand() & 0xFF) / 10.0f;
    }
    // initialData(h_B, nElem);
    // generate different seed for random number
    srand((unsigned)time(&t));
    for (int i = 0; i < nElem; i++) {
      h_B[i] = (float)(rand() & 0xFF) / 10.0f;
    }

    std::vector<KernelArg> args;
    args.emplace_back(KernelArg{h_A, nBytes, false});
    args.emplace_back(KernelArg{h_B, nBytes, false});
    args.emplace_back(KernelArg{gpuRef, nBytes, true});

    dim3 block(32, 8, 2);
    int blockSize = block.x * block.y * block.z;
    dim3 grid((nElem + blockSize - 1) / blockSize);
    Kernel k =
        source.program("sumArrayOnGPU").compile(options).configure(grid, block);
    k.launch(args, device);

    // add vector at host side for result checks
    for (int idx = 0; idx < nElem; idx++)
      hostRef[idx] = h_A[idx] + h_B[idx];

    // check device results
    double epsilon = 1.0E-8;
    bool match = 1;
    for (int i = 0; i < nElem; ++i) {
      if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
        match = 0;
        char *error_string;
        asprintf(
            &error_string,
            "Arrays do not match! \n->host %5.2f gpu %5.2f at current %d\n",
            hostRef[i], gpuRef[i], i);
        FAIL(error_string);
        break;
      }
    }
  }

  delete[] h_A;
  delete[] h_B;
  delete[] hostRef;
  delete[] gpuRef;
}

TEST_CASE("sumArray (size of array as an argument)", "[example_program]") {
  Device device = Devices::findDevice();
  Options options{yacx::options::GpuArchitecture(device),
                  yacx::options::FMAD(false)};
  options.insert("--std", "c++14");
  Source source{
      "extern \"C\" __global__ void sumArrayOnGPU(float *A, float *B, float* "
      "C, int* size){\n"
      "int "
      "i_inBlock=threadIdx.x+threadIdx.y*blockDim.x+threadIdx.z*blockDim.y*"
      "blockDim.x;\n"
      "int blockID= blockIdx.x;\n"
      "int i = i_inBlock + blockID*(blockDim.x*blockDim.y*blockDim.z);\n"
      "if(i<=*size){"
      "C[i]=A[i]+B[i];\n"
      "}\n"
      "}\n"};

  // set up data size of vectors
  int nElem = GENERATE(1, 32, 48, 128, 1024, 2028, 10240, 42000, 100000);
  // malloc host memory
  size_t nBytes = nElem * sizeof(float);
  float *h_A, *h_B, *hostRef, *gpuRef;
  h_A = new float[nBytes];
  h_B = new float[nBytes];
  hostRef = new float[nBytes];
  gpuRef = new float[nBytes];

  SECTION("SUM_ARRAY WITH random values and with different sizes") {
    // initialize data at host side
    // initialData(h_A, nElem);
    // generate different seed for random number
    time_t t;
    srand((unsigned)time(&t));
    for (int i = 0; i < nElem; i++) {
      h_A[i] = (float)(rand() & 0xFF) / 10.0f;
    }
    // initialData(h_B, nElem);
    // generate different seed for random number
    srand((unsigned)time(&t));
    for (int i = 0; i < nElem; i++) {
      h_B[i] = (float)(rand() & 0xFF) / 10.0f;
    }

    std::vector<KernelArg> args;
    args.emplace_back(KernelArg{h_A, nBytes, false});
    args.emplace_back(KernelArg{h_B, nBytes, false});
    args.emplace_back(KernelArg{gpuRef, nBytes, true});
    args.emplace_back(KernelArg{&nElem, sizeof(int), false});

    constexpr int warpsize = 32;
    int max_block_DIM = 0;
    CUDA_SAFE_CALL(cuDeviceGetAttribute(
        (int *)&max_block_DIM, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
        device.get()));
    const int grid_x = (nElem + max_block_DIM - 1) / max_block_DIM;
    const int block_y = (nElem / grid_x + warpsize - 1) / warpsize;

    dim3 block(warpsize, block_y, 1);
    dim3 grid(grid_x);

    Logger(yacx::loglevel::DEBUG1)
        << "Program sumArray compiles with " << nElem << " Threads"
        << "\n";
    Kernel k =
        source.program("sumArrayOnGPU").compile(options).configure(grid, block);
    Logger(yacx::loglevel::DEBUG1)
        << "Program sumArray started with " << nElem << " Threads"
        << "\n";
    k.launch(args, device);

    // add vector at host side for result checks
    for (int idx = 0; idx < nElem; idx++)
      hostRef[idx] = h_A[idx] + h_B[idx];

    // check device results
    double epsilon = 1.0E-8;
    bool match = 1;
    for (int i = 0; i < nElem; ++i) {
      if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
        match = 0;
        char *error_string;
        asprintf(
            &error_string,
            "Arrays do not match! \n->host %5.2f gpu %5.2f at current %d\n",
            hostRef[i], gpuRef[i], i);
        FAIL(error_string);
        break;
      }
    }
  }

  delete[] h_A;
  delete[] h_B;
  delete[] hostRef;
  delete[] gpuRef;
}