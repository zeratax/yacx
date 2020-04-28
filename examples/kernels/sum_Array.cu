#include "cuda_runtime.h"

template<typename type>
__global__ void sumArrayOnGPU(type *A,type *B, type *C) {
    int i_inBlock = threadIdx.x + threadIdx.y * blockDim.x +threadIdx.z * blockDim.y * blockDim.x;
    int blockID = blockIdx.x;
    int i = i_inBlock + blockID * (blockDim.x * blockDim.y* blockDim.z);
    C[i] = A[i] + B[i];
}