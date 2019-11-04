extern "C" __global__ void sumArrayOnGPU(float *A, float *B, float *C) {
  int i_inBlock = threadIdx.x + threadIdx.y * blockDim.x +
                  threadIdx.z * blockDim.y * blockDim.x;
  int blockID = blockIdx.x;
  int i = i_inBlock + blockID * (blockDim.x * blockDim.y * blockDim.z);
  C[i] = A[i] + B[i];
}