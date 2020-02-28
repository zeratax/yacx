#pragma once

#include <nvfunctional>

// Performs a warp-wide reduction with func as the reduction function while ensuring that inactive threads, 
// whose shuffle value is always zero, are being ignored during the reduction as these might distort the
// result for certain reduction functions (e.g. minimum or multiplication).
__inline__ __device__ 
long long warpReduce(long long val, const nvstd::function<long long(long long, long long)> &func, int N) {
	
	// calculate the absolute ID of this thread
	int ID = blockIdx.x * blockDim.x + threadIdx.x;
	
	// repeat the shuffle operation log2(32) = 5 times
	for (int offset = warpSize / 2; offset > 0; offset /= 2) {
		// calculate the source lane ID of the next shuffle instruction
		int shuffle_id = ID + offset;
		// perform the shuffle operation
		long long x = __shfl_down_sync(0xFFFFFFFF, val, offset);
		// ignore the read value if the source lane is out of bounds (so inactive)
		val = (shuffle_id < N) ? func(val, x) : val;
	}
	
	return val;
}