#include "examples/kernels/value.hpp"

template<typename type>
__global__ void sumArrayOnGPUWithHeader(type *A,type *B, type *C) {
    value value_default;

    int i_inBlock = threadIdx.x + threadIdx.y * blockDim.x +threadIdx.z * blockDim.y * blockDim.x;
    int blockID = blockIdx.x;
    int i = i_inBlock + blockID * (blockDim.x * blockDim.y* blockDim.z);
    C[i] = A[i] + B[i];

    if (C[i]>=5.0f)
     C[i]+=value_default.float_default;
}