template <int TILE_WIDTH>
__global__ void MatrixMultiplicationFast(float *Md, float *Nd, float *Pd,
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
      for (int n{0}; n < 4; n++) {
        sum[n] += Ml[row_l + n][k] * Nl[k][col_l];
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int n{0}; n < 4; n++) {
    Pd[(row + n) * width + col] = sum[n];
  }
}

extern "C" __global__ void MatrixMultiplication(float *A, float *B, float *C,
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

template <int BLOCK_SIZE>
__global__ void multiply(float *left, float *right, float *res, int dim) {

  int i,j;
  float temp = 0;

  __shared__ float Left_shared_t [BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float Right_shared_t[BLOCK_SIZE][BLOCK_SIZE];

  // Row i of matrix left
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;


  for (int tileNUM = 0; tileNUM < gridDim.x; tileNUM++) {

    // Column j of matrix left
    j = tileNUM * BLOCK_SIZE + threadIdx.x;
    i = tileNUM * BLOCK_SIZE + threadIdx.y;
    // Load left[i][j] to shared mem

    Left_shared_t[threadIdx.y][threadIdx.x] = left[row * dim + j];// Coalesced access
    // Load right[i][j] to shared mem

    Right_shared_t[threadIdx.y][threadIdx.x] = right[i * dim + col]; // Coalesced access
    // Synchronize before computation
    __syncthreads();

    // Accumulate one tile of res from tiles of left and right in shared mem
    for (int k = 0; k < BLOCK_SIZE; k++) {

      temp += Left_shared_t[threadIdx.y][k] * Right_shared_t[k][threadIdx.x]; //no shared memory bank conflict
    }
    // Synchronize
    __syncthreads();
  }
  // Store accumulated value to res
  res[row * dim + col] = temp;
}
