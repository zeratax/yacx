#include "examples/kernels/gauss.h"

extern "C" __global__ void gaussFilterKernel(Pixel *image,
                                             float weight[5][5],
                                             int width, int height) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  Pixel new_value;
  new_value.r = 0;
  new_value.g = 0;
  new_value.b = 0;
  for (int xl = -2; xl <= 2; ++xl) {
    for (int yl = -2; yl <= 2; ++yl) {
      if (((col + xl) + (row + yl) * width) < 0 ||
          ((col + xl) + (row + yl) * width) >= width * height) {
        continue;
      }
      new_value.r +=
          image[(col + xl) + (row + yl) * width].r * weight[xl + 2][yl + 2];
      new_value.g +=
          image[(col + xl) + (row + yl) * width].g * weight[xl + 2][yl + 2];
      new_value.b +=
          image[(col + xl) + (row + yl) * width].b * weight[xl + 2][yl + 2];
    }
  }
  image[col + row * width] = new_value;
}