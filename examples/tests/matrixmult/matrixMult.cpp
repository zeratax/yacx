#include <builtin_types.h>
#include <cstdio>
#include <cuda.h>
#include <nvrtc.h>
#include <string>
#include <sys/time.h>
#include <time.h>

CUdevice device;
CUcontext context;
CUmodule module;
CUfunction function;
const int WIDTH = 1024;

#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

inline void __checkCudaErrors(CUresult err, const char *file, const int line) {
  if (CUDA_SUCCESS != err) {
    fprintf(stderr, "CUDA Driver API error = %04d from file <%s>, line %i.\n",
            err, file, line);
    exit(-1);
  }
}

void checkResult(float *hostRef, float *gpuRef, const int N) {
  double epsilon{1.0E-8};
  bool match{false};
  for (int i{0}; i < N; ++i) {
    if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
      match = 0;
      printf("Arrays do not match!\n");
      printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
      break;
    }
  }
  if (match) {
    printf("Arrays match\n\n");
  }
}

// fill f width size many random float values
void fill(float *f, size_t size) {
  srand(time(NULL));
  for (size_t i = 0; i < size; i += 1)
    f[i] = (static_cast<float>(rand())) / RAND_MAX;
}

// compares every pair lhs[i] and rhs[i] for i < width
void compare(float *lhs, float *rhs, int width) {
  int errors{0};
  for (int i{0}; i < width; i += 1) {
    if ((lhs[i] - rhs[i]) != 0) {
      printf("%f : %f\n", lhs[i], rhs[i]);
      errors += 1;
    }
  }
  if (errors > 0)
    printf("%d errors occured.\n", errors);
  else
    printf("no errors occured.\n");
}

// sequentiell matrix multiplication
void MatrixMulSeq(const float *M, const float *N, float *P, size_t width) {
  size_t Col, Row, k;
  for (Col = 0; Col < width; ++Col)
    for (Row = 0; Row < width; ++Row) {
      float sum = 0;
      for (k = 0; k < width; k += 1) {
        sum += M[Row * width + k] * N[k * width + Col];
      }
      P[Row * width + Col] = sum;
    }
}

void ladeModulUndFunktion(const CUcontext &context, CUmodule *module,
                          CUfunction *function, const std::string &module_file,
                          const std::string &kernel_name) {

  CUresult err = cuModuleLoad(module, module_file.c_str());
  if (err != CUDA_SUCCESS) {
    printf("%i\n", err);
    fprintf(stderr, "* Error loading the module %s\n",
            const_cast<char *>(module_file.c_str()));
    cuCtxDestroy(context);
    exit(-1);
  }

  err = cuModuleGetFunction(function, *module, kernel_name.c_str());

  if (err != CUDA_SUCCESS) {
    fprintf(stderr, "* Error getting kernel function %s\n",
            kernel_name.c_str());
    cuModuleUnload(*module);
    exit(-1);
  }
}

void initCUDA(CUdevice *device, CUcontext *context, size_t *totalGlobalMem) {
  int deviceCount{0};
  CUresult err = cuInit(CUDA_SUCCESS);
  int major;
  int minor;

  if (err == CUDA_SUCCESS) {
    checkCudaErrors(cuDeviceGetCount(&deviceCount));
  }
  if (deviceCount == 0) {
    printf("Error: no devices supporting CUDA\n");
    exit(-1);
  }
  checkCudaErrors(cuDeviceGet(device, 0));

  char name[50];
  cuDeviceGetName(name, 50, *device);
  printf("> Using device 0: %s\n", name);

  checkCudaErrors(cuDeviceGetAttribute(
      &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, *device));
  checkCudaErrors(cuDeviceGetAttribute(
      &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, *device));
  printf("> GPU Device has SM %d.%d compute capability\n", major, minor);

  checkCudaErrors(cuDeviceTotalMem(totalGlobalMem, *device));
  printf("  Total amount of global memory:   %llu bytes\n",
         reinterpret_cast<unsigned long long>(totalGlobalMem));

  // Erstelle den Kontext
  // err = cuDevicePrimaryCtxRetain(&context,device);
  err = cuCtxCreate(context, 0, *device);
  if (err != CUDA_SUCCESS) {
    fprintf(stderr, "* Error initializing the CUDA context.\n");
    // cuDevicePrimaryCtxRelease(device);
    cuCtxDestroy(*context);
    exit(-1);
  }
}

void initModule(const CUcontext &context, CUmodule *module,
                CUfunction *function) {
  std::string nameModule("matrixMult");
  std::string module_file = nameModule;
  module_file.append(".cubin");
  std::string kernel_name("MatrixMultKernel");

  // kompiliereCubinFile(nameModule.c_str());
  ladeModulUndFunktion(context, module, function,
                       (const std::string)module_file,
                       (const std::string)kernel_name);
}

void MatrixMulCUDA(const float *h_M, const float *h_N, float *h_P,
                   size_t width) {
  CUdeviceptr d_M, d_N, d_P;
  const size_t matrix_size = WIDTH * WIDTH * sizeof(float);
  checkCudaErrors(cuMemAlloc(&d_M, matrix_size));
  checkCudaErrors(cuMemAlloc(&d_N, matrix_size));
  checkCudaErrors(cuMemAlloc(&d_P, matrix_size));
  checkCudaErrors(cuMemcpyHtoD(d_M, &h_M, matrix_size));
  checkCudaErrors(cuMemcpyHtoD(d_N, &h_N, matrix_size));

  dim3 block(16, 16 / 4);
  dim3 grid(WIDTH, WIDTH / 4);

  void *args[]{&d_M, &d_N, &d_P, &width};
  cuLaunchKernel(function, grid.x, grid.y, grid.z, // grid
                 block.x, block.y, block.z,        // block
                 0, 0, args, 0);

  // copy kernel result back to host side
  cuMemcpyDtoH(&h_P, d_P, matrix_size);
}

int main(void) {
  struct timeval start, end;

  float M[WIDTH];
  float N[WIDTH];
  float P_cuda[WIDTH];
  float P_seq[WIDTH];

  fill(M, WIDTH * WIDTH);
  fill(N, WIDTH * WIDTH);

  size_t totalGlobalMem;
  // initialisiere das Device und den Context
  initCUDA(&device, &context, &totalGlobalMem);
  // Module und Funktion werden geladen
  initModule(context, &module, &function);

  gettimeofday(&start, NULL);
  MatrixMulCUDA(M, N, P_cuda, WIDTH);
  gettimeofday(&end, NULL);
  printf("Time elapsed CUDA: %fmsecs\n",
         static_cast<float>(1000.0 * (end.tv_sec - start.tv_sec) +
                            0.001 * (end.tv_usec - start.tv_usec)));

  gettimeofday(&start, NULL);
  MatrixMulSeq(M, N, P_seq, WIDTH);
  gettimeofday(&end, NULL);
  printf("Time elapsed Seq: %fmsecs\n",
         static_cast<float>(1000.0 * (end.tv_sec - start.tv_sec) +
                            0.001 * (end.tv_usec - start.tv_usec)));

  compare(P_seq, P_cuda, WIDTH * WIDTH);

  return 0;
}
