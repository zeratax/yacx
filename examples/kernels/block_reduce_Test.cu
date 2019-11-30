#pragma once

#include "/tmp/tmp.cTlciDtCIj/examples/kernels/block_reduce.h"

extern "C" __global__
void blockReduceSumTest(long long val, long *dst, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= n) return;

    dst[i] = blockReduceSum(i);
}