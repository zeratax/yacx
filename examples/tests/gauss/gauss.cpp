#include <builtin_types.h>
#include <cmath>
#include <cstdio>
#include <cuda.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nvrtc.h>
#include <stdlib.h>
#include <string>
#include <sys/time.h>
#include <time.h>

#include "gauss.h"

CUdevice device;
CUcontext context;
CUmodule module;
CUfunction function;

#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

inline void __checkCudaErrors(CUresult err, const char *file, const int line) {
  if (CUDA_SUCCESS != err) {
    fprintf(stderr, "CUDA Driver API error = %04d from file <%s>, line %i.\n",
            err, file, line);
    exit(-1);
  }
}

void writePPM(Pixel *pixels, const char *filename, int width, int height) {
  std::ofstream outputFile(filename, std::ios::binary);

  // write header:
  outputFile << "P6\n" << width << " " << height << "\n255\n";

  outputFile.write(reinterpret_cast<const char *>(pixels),
                   sizeof(Pixel) * width * height);
}

// Pointer returned must be explicitly freed!
Pixel *readPPM(const char *filename, int *width, int *height) {
  std::ifstream inputFile(filename, std::ios::binary);

  // parse harder
  // first line: P6\n
  inputFile.ignore(2, '\n'); // ignore P6
  // possible comments:
  while (inputFile.peek() == '#') {
    inputFile.ignore(1024, '\n');
  } // skip comment
  // next line: width_height\n
  inputFile >> (*width);
  inputFile.ignore(1, ' '); // ignore space
  inputFile >> (*height);
  inputFile.ignore(1, '\n'); // ignore newline
  // possible comments:
  while (inputFile.peek() == '#') {
    inputFile.ignore(1024, '\n');
  } // skip comment
  // last header line: 255\n:
  inputFile.ignore(3, '\n'); // ignore 255 and newline

  Pixel *data =
      static_cast<Pixel *>(malloc(sizeof(Pixel) * (*width) * (*height)));

  inputFile.read(reinterpret_cast<char *>(data),
                 sizeof(Pixel) * (*width) * (*height));

  return data;
}

using std::string;

void calculateWeights(float weights[5][5]) {
  float sigma = 1.0;
  float r, s = 2.0 * sigma * sigma;

  // sum is for normalization
  float sum = 0.0;

  // generate weights for 5x5 kernel
  for (int x = -2; x <= 2; x++) {
    for (int y = -2; y <= 2; y++) {
      r = x * x + y * y;
      weights[x + 2][y + 2] = exp(-(r / s)) / (M_PI * s);
      sum += weights[x + 2][y + 2];
    }
  }

  // normalize the weights
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      weights[i][j] /= sum;
    }
  }
}

void gaussFilterseq(Pixel *image, int width, int height, float weight[5][5]) {
  for (int x = 0; x < width; ++x) {
    for (int y = 0; y < height; ++y) {

      Pixel new_value;
      new_value.r = 0;
      new_value.g = 0;
      new_value.b = 0;
      for (int xl = -2; xl <= 2; ++xl) {
        for (int yl = -2; yl <= 2; ++yl) {
          if (((x + xl) + (y + yl) * width) < 0 ||
              ((x + xl) + (y + yl) * width) >= width * height) {
            continue;
          }
          new_value.r +=
              image[(x + xl) + (y + yl) * width].r * weight[xl + 2][yl + 2];
          new_value.g +=
              image[(x + xl) + (y + yl) * width].g * weight[xl + 2][yl + 2];
          new_value.b +=
              image[(x + xl) + (y + yl) * width].b * weight[xl + 2][yl + 2];
        }
      }
      image[x + y * width] = new_value;
    }
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
  std::string nameModule("gauss");
  std::string module_file = nameModule;
  module_file.append(".cubin");
  std::string kernel_name("gaussFilterKernel");

  // kompiliereCubinFile(nameModule.c_str());
  ladeModulUndFunktion(context, module, function,
                       (const std::string)module_file,
                       (const std::string)kernel_name);
}

void gaussFilterCUDA(Pixel *h_image, int width, int height,
                     float h_weights[5][5]) {
  CUdeviceptr d_image, d_weights;
  size_t size_pixel = height * width * sizeof(Pixel);
  size_t size_weights = 5 * 5 * sizeof(float);
  checkCudaErrors(cuMemAlloc(&d_image, size_pixel));
  checkCudaErrors(cuMemAlloc(&d_weights, size_weights));
  checkCudaErrors(cuMemcpyHtoD(d_image, &h_image, size_pixel));
  checkCudaErrors(cuMemcpyHtoD(d_weights, &h_weights, size_weights));

  dim3 block(1, 1, 1);
  dim3 grid(width, height, 1);

  void *args[]{&d_image, &d_weights, &width, &height};
  cuLaunchKernel(function, grid.x, grid.y, grid.z, // grid
                 block.x, block.y, block.z,        // block
                 0, 0, args, 0);

  // copy kernel result back to host side
  cuMemcpyDtoH(&h_image, d_image, size_pixel);
}

int main(int argc, char **argv) {
  struct timeval start, end;

  const char *inFilename = (argc > 1) ? argv[1] : "lena.ppm";
  const char *outFilename_seq = (argc > 2) ? argv[2] : "output_seq.ppm";
  const char *outFilename_cuda = (argc > 2) ? argv[2] : "output_cuda.ppm";

  float weights[5][5];
  calculateWeights(weights);
  int width;
  int height;

  Pixel *image_seq = readPPM(inFilename, &width, &height);
  Pixel *image_cuda = readPPM(inFilename, &width, &height);
  size_t totalGlobalMem;
  initCUDA(&device, &context, &totalGlobalMem);
  initModule(context, &module, &function);

  gettimeofday(&start, NULL);
  gaussFilterCUDA(image_cuda, width, height, weights);
  gettimeofday(&end, NULL);
  printf("Time elapsed CUDA: %fmsecs\n",
         static_cast<float>(1000.0 * (end.tv_sec - start.tv_sec) +
                            0.001 * (end.tv_usec - start.tv_usec)));

  gettimeofday(&start, NULL);
  gaussFilterseq(image_seq, width, height, weights);
  gettimeofday(&end, NULL);
  printf("Time elapsed seq: %fmsecs\n",
         static_cast<float>(1000.0 * (end.tv_sec - start.tv_sec) +
                            0.001 * (end.tv_usec - start.tv_usec)));

  writePPM(image_seq, outFilename_seq, width, height);
  writePPM(image_cuda, outFilename_cuda, width, height);
  free(image_seq);  // must be explicitly freed
  free(image_cuda); // must be explicitly freed

  return 0;
}
