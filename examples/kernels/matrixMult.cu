template <int TILE_WIDTH>
__global__ void MatrixMulty1unfolded(float *Md, float *Nd, float *Pd,
                                     int width) {
  const int row{4 * blockIdx.y * blockDim.y + threadIdx.y};
  const int col{blockIdx.x * blockDim.x + threadIdx.x};
  const int row_l{4 * threadIdx.y};
  const int col_l{threadIdx.x};
  __shared__ float Ml[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Nl[TILE_WIDTH][TILE_WIDTH];

  float sum0{0.0f};
  float sum1{0.0f};
  float sum2{0.0f};
  float sum3{0.0f};
  for (int m{0}; m < width; m += TILE_WIDTH) {
    __syncthreads();
    Ml[row_l][col_l] = Md[(row)*width + m + col_l];
    Ml[row_l + 1][col_l] = Md[(row + 1) * width + m + col_l];
    Ml[row_l + 2][col_l] = Md[(row + 2) * width + m + col_l];
    Ml[row_l + 3][col_l] = Md[(row + 3) * width + m + col_l];

    Nl[row_l][col_l] = Nd[width * (m + row_l) + col];
    Nl[row_l + 1][col_l] = Nd[width * (m + 1 + row_l) + col];
    Nl[row_l + 2][col_l] = Nd[width * (m + 2 + row_l) + col];
    Nl[row_l + 3][col_l] = Nd[width * (m + 3 + row_l) + col];

    __syncthreads();
    for (int k = 0; k < TILE_WIDTH; ++k) {
      sum0 += Ml[row_l][k] * Nl[k][col_l];
      sum1 += Ml[row_l + 1][k] * Nl[k][col_l];
      sum2 += Ml[row_l + 2][k] * Nl[k][col_l];
      sum3 += Ml[row_l + 3][k] * Nl[k][col_l];
    }
    __syncthreads();
  }
  Pd[row * width + col] = sum0;
  Pd[(row + 1) * width + col] = sum1;
  Pd[(row + 2) * width + col] = sum2;
  Pd[(row + 3) * width + col] = sum3;
}

template <int TILE_WIDTH, int GRANULARITY>
__global__ void MatrixMulty1(float *Md, float *Nd, float *Pd, int width) {
  const int col{blockIdx.x * blockDim.x + threadIdx.x};
  const int row{GRANULARITY * blockIdx.y * blockDim.y + threadIdx.y};
  const int col_l{threadIdx.x};
  const int row_l{GRANULARITY * threadIdx.y};
  __shared__ float Ml[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Nl[TILE_WIDTH][TILE_WIDTH];

  float sum[GRANULARITY];
#pragma unroll
  for (int n{0}; n < GRANULARITY; ++n) {
    sum[n] = 0.0f;
  }

  for (int m{0}; m < width; m += TILE_WIDTH) {
#pragma unroll
    for (int n{0}; n < GRANULARITY; ++n) {
      Ml[row_l + n][col_l] = Md[(row + n) * width + (m + col_l)];
    }
#pragma unroll
    for (int n{0}; n < GRANULARITY; ++n) {
      Nl[row_l + n][col_l] = Nd[(m + row_l + n) * width + col];
    }

    __syncthreads();
#pragma unroll
    for (int k{0}; k < TILE_WIDTH; ++k) {
#pragma unroll
      for (int n{0}; n < GRANULARITY; ++n) {
        sum[n] += Ml[row_l + n][k] * Nl[k][col_l];
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int n{0}; n < GRANULARITY; ++n) {
    Pd[(row + n) * width + col] = sum[n];
  }
}

// see:
// https://github.com/kberkay/Cuda-Matrix-Multiplication/blob/master/matrix_Multiplication.cu#L100
template <int BLOCK_SIZE>
__global__ void MatrixMulty2(float *left, float *right, float *res, int dim) {

  int i, j;
  float temp = 0;

  __shared__ float Left_shared_t[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float Right_shared_t[BLOCK_SIZE][BLOCK_SIZE];

  // Row i of matrix left
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  for (int tileNUM = 0; tileNUM < gridDim.x; tileNUM++) {

    // Column j of matrix left
    j = tileNUM * BLOCK_SIZE + threadIdx.x;
    i = tileNUM * BLOCK_SIZE + threadIdx.y;
    // Load left[i][j] to shared mem

    Left_shared_t[threadIdx.y][threadIdx.x] =
        left[row * dim + j]; // Coalesced access
    // Load right[i][j] to shared mem

    Right_shared_t[threadIdx.y][threadIdx.x] =
        right[i * dim + col]; // Coalesced access
    // Synchronize before computation
    __syncthreads();

    // Accumulate one tile of res from tiles of left and right in shared mem
    for (int k = 0; k < BLOCK_SIZE; k++) {

      temp += Left_shared_t[threadIdx.y][k] *
              Right_shared_t[k][threadIdx.x]; // no shared memory bank conflict
    }
    // Synchronize
    __syncthreads();
  }
  // Store accumulated value to res
  res[row * dim + col] = temp;
}

extern "C" __global__ void MatrixMultyNaive(float *A, float *B, float *C,
                                            int N) {

  const int ROW = blockIdx.y * blockDim.y + threadIdx.y;
  const int COL = blockIdx.x * blockDim.x + threadIdx.x;

  float sum{0};

  if (ROW < N && COL < N) {
    for (auto i{0}; i < N; i++) {
      sum += A[ROW * N + i] * B[i * N + COL];
    }
  }
  C[ROW * N + COL] = sum;
}
