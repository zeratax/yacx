const int TILE_WIDTH{128};

extern "C" __global__ void MatrixMultKernel(float *Md, float *Nd, float *Pd,
                                            int width) {
  int row = 4 * blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row_l = 4 * threadIdx.x;
  int col_l = threadIdx.y;
  __shared__ float Ml[16][16];
  __shared__ float Nl[16][16];

  float sum[4]{0.0f, 0.0f, 0.0f, 0.0f};
  for (int m = 0; m < width; m += TILE_WIDTH) {
    #pragma unroll
    for (int n = 0; n < 4; n++) {
      Ml[row_l + n][col_l] = Md[(row + n) * width + m + col_l];
    }


    #pragma unroll
    for (int n = 0; n < 4; n++) {
      Nl[row_l + n][col_l] = Nd[width * (m + n + row_l) + col];
    }

    __syncthreads();
    for (int k = 0; k < TILE_WIDTH; k++) {
      #pragma unroll
      for (int n = 0; n < 4; n++) {
        sum[n] += Ml[row_l + n][k] * Nl[k][col_l];
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for (int n{0}; n < 4; n++) {
    Pd[(row + n) * width + col] = sum[4];
  }
}
