#include "kernels/block_reduce.h"

extern "C" __global__
void deviceReduceKernel(long* in, long* out, int N) {
    long sum = 0;

    //reduce multiple elements per thread
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        sum += in[i];
    }

    sum = blockReduceSum(sum);
    if (threadIdx.x == 0) out[blockIdx.x] = sum;
}
