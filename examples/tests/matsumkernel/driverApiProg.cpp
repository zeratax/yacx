// Copyright 2019 André Hodapp
#include <stdio.h>
#include <stdlib.h>

#include <builtin_types.h> // z.B. für dim3
#include <cuda.h>

// C++ Headers
#include <string>

// This will output the proper CUDA error strings
// in the event that a CUDA host call returns an error
#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

inline void __checkCudaErrors(CUresult err, const char *file, const int line) {
  if (CUDA_SUCCESS != err) {
    fprintf(stderr, "CUDA Driver API error = %04d from file <%s>, line %i.\n",
            err, file, line);
    exit(-1);
  }
}

void checkResult(float *hostRef, float *gpuRef, const int N) {
  double epsilon = 1.0E-8;
  bool match = 1;
  for (int i = 0; i < N; ++i) {
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

void initialData(float *ip, int size) {
  // generate different seed for random number
  time_t t;
  srand((unsigned)time(&t));
  for (int i = 0; i < size; i++) {
    ip[i] = static_cast<float>((rand() & 0xFF) / 10.0f);
  }
}
void sumArraysOnHost(float *A, float *B, float *C, const int N) {
  for (int idx = 0; idx < N; idx++)
    C[idx] = A[idx] + B[idx];
}

//_________________________________________________________________
// Unterfunktionen
void bestimmeBlockUndGridDimension(const CUdevice &device, dim3 block,
                                   dim3 grid);
void ladeModulUndFunktion(const CUcontext &context, CUmodule *module,
                          CUfunction *function, const std::string &module_file,
                          const std::string &kernel_name);
void kompiliereCubinFile(std::string nameFuc);

// Hauptfunktionen
void initCUDA(CUdevice *device, CUcontext *context, size_t *totalGlobalMem);
void initModule(const CUcontext &context, CUmodule *module,
                CUfunction *function);

int main(void) {
  // set up device
  CUdevice device;
  CUcontext context;
  CUmodule module;
  CUfunction function;
  size_t totalGlobalMem;
  // initialisiere das Device und den Context
  initCUDA(&device, &context, &totalGlobalMem);
  // Module und Funktion werden geladen
  initModule(context, &module, &function);

  // bis hierhin wird der Cuda Kontext und die Module initialisiert

  // set up data size of vectors
  int nElem = 1024;
  printf("Vector size %d\n", nElem);

  // malloc host memory
  size_t nBytes = nElem * sizeof(float);

  float *h_A, *h_B, *hostRef, *gpuRef;
  h_A = static_cast<float *>(malloc(nBytes));
  h_B = static_cast<float *>(malloc(nBytes));
  hostRef = static_cast<float *>(malloc(nBytes));
  gpuRef = static_cast<float *>(malloc(nBytes));

  // initialize data at host side
  initialData(h_A, nElem);
  initialData(h_B, nElem);
  printf("nBytes: %u\n", (unsigned int)nBytes);
  // malloc device global memory
  CUdeviceptr d_A, d_B,
      d_C; // cuMemAlloc(d_A) mit d_A als Pointer funktioniert nicht
  checkCudaErrors(cuMemAlloc(&d_A, (unsigned int)nBytes));
  checkCudaErrors(cuMemAlloc(&d_B, nBytes));
  checkCudaErrors(cuMemAlloc(&d_C, nBytes));

  // transfer data from host to device
  checkCudaErrors(cuMemcpyHtoD(d_A, static_cast<void *>(h_A), nBytes));
  checkCudaErrors(cuMemcpyHtoD(d_B, static_cast<void *>(h_B), nBytes));

  // invoke kernel at host sidecuMemFreecuMemFree
  dim3 block(32, 8, 2);
  int blockSize = block.x * block.y * block.z;
  dim3 grid((nElem + blockSize - 1) / blockSize);
  void *args[3] = {&d_A, &d_B, &d_C};
  checkCudaErrors(cuLaunchKernel(function, grid.x, grid.y, grid.z, // grid
                                 block.x, block.y, block.z,        // block
                                 0, 0, args, 0));
  // sumArrayOnGPU<<<grid, block>>> (d_A, d_B, d_C);
  printf("Execution configuration <<<(%d,%d,%d),(%d,%d,%d)>>>\n", grid.x,
         grid.y, grid.z, block.x, block.y, block.z);

  // copy kernel result back to host side
  checkCudaErrors(cuMemcpyDtoH(gpuRef, d_C, nBytes));

  // add vector at host side for result checks
  sumArraysOnHost(h_A, h_B, hostRef, nElem);

  // check device results
  checkResult(hostRef, gpuRef, nElem);

  // free device global memory
  cuMemFree(d_A);
  cuMemFree(d_B);
  cuMemFree(d_C);

  // free host memory
  free(h_A);
  free(h_B);
  free(hostRef);
  free(gpuRef);

  // free module and context
  cuModuleUnload(module);
  cuCtxDestroy(context);
  // cuDevicePrimaryCtxRelease(device); //hiermit ergeben sich Fehler in
  // Kombination mit cuModuleLoad

  return 0;
}

void initCUDA(CUdevice *device, CUcontext *context, size_t *totalGlobalMem) {
  int deviceCount = 0;
  // Initializes the driver API and
  // must be called before any other function from the driver API
  CUresult err = cuInit(CUDA_SUCCESS);
  int major;
  int minor;

  if (err == CUDA_SUCCESS) {
    /*
     *Gibt die Anzahl der Devices mit compute caoability 2.0 oder größer an
     */
    checkCudaErrors(cuDeviceGetCount(&deviceCount));
  }
  if (deviceCount == 0) {
    printf("Error: no devices supporting CUDA\n");
    exit(-1);
  }
  // Returns a handle to a compute device.
  checkCudaErrors(cuDeviceGet(device, 0));

  char name[50];
  // Returns an identifer string for the device.
  cuDeviceGetName(name, 50, *device);
  printf("> Using device 0: %s\n", name);

  checkCudaErrors(cuDeviceGetAttribute(
      &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, *device));
  checkCudaErrors(cuDeviceGetAttribute(
      &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, *device));
  printf("> GPU Device has SM %d.%d compute capability\n", major, minor);

  dim3 block, grid;
  bestimmeBlockUndGridDimension(*device, block, grid);
  printf("block.x = %i\n", block.x);

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
  std::string nameModule("sumArrayOnDevice");
  std::string module_file = nameModule;
  module_file.append(".cubin");
  std::string kernel_name("sumArrayOnGPU");

  kompiliereCubinFile(nameModule.c_str());
  ladeModulUndFunktion(context, module, function,
                       (const std::string)module_file,
                       (const std::string)kernel_name);
}

// Unterfunktionen
void bestimmeBlockUndGridDimension(const CUdevice &device, dim3 block,
                                   dim3 grid) {
  checkCudaErrors(cuDeviceGetAttribute(reinterpret_cast<int *>(block.x),
                                       CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
                                       device));
  checkCudaErrors(cuDeviceGetAttribute(reinterpret_cast<int *>(block.y),
                                       CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
                                       device));
  checkCudaErrors(cuDeviceGetAttribute(reinterpret_cast<int *>(block.z),
                                       CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z,
                                       device));
  checkCudaErrors(cuDeviceGetAttribute(reinterpret_cast<int *>(grid.x),
                                       CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
                                       device));
  checkCudaErrors(cuDeviceGetAttribute(reinterpret_cast<int *>(grid.y),
                                       CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,
                                       device));
  checkCudaErrors(cuDeviceGetAttribute(reinterpret_cast<int *>(grid.z),
                                       CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z,
                                       device));
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
    // cuDevicePrimaryCtxRelease(device);
    exit(-1);
  }

  err = cuModuleGetFunction(function, *module, kernel_name.c_str());

  if (err != CUDA_SUCCESS) {
    fprintf(stderr, "* Error getting kernel function %s\n",
            kernel_name.c_str());
    cuModuleUnload(*module);
    // cuDevicePrimaryCtxRelease(device);
    exit(-1);
  }
}

void kompiliereCubinFile(std::string nameModule) {
  nameModule.append(".cu");
  std::string befehl("nvcc -arch=sm_30 -Xptxas= -v --cubin ");
  befehl.append(nameModule.c_str());
  printf("\n_________________Kompilieren von .cubin_________________\n");
  int status = system(befehl.c_str());
  printf("Status nach .cubin kompilieren ist %i \n", status);
  printf("\n_____________Kompilieren von .cubin zu Ende_____________\n");
}
