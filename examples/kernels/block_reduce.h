#pragma once

#include "warp_reduce.h"

// Performs a block-wide reduction with func as the reduction function while ensuring that inactive threads, 
// whose shuffle value is always zero, are being ignored during the reduction as these might distort the
// result for certain reduction functions (e.g. minimum or multiplication).
// This block-wide reduction is being done by performing a reduction for each warp in this block,
// before reducing the results of each warp within the first warp of this block.
__inline__ __device__ 
long long blockReduce(long long val, const nvstd::function<long long(long long, long long)> &func, int N) {

	// Shared mem for 32 partial sums
	static __shared__ long long shared[32];
	
	// Calculate the lane and warp ID
	int lane = threadIdx.x % warpSize;
	int wid = threadIdx.x / warpSize;

	// Each warp performs partial reduction
	val = warpReduce(val, func, N); 

	// Write reduced value to shared memory
	if (lane == 0) shared[wid] = val;

	// Wait for all partial reductions
	__syncthreads();

	// Read from shared memory only if that warp existed
	val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

	// Check if this thread is in the last active block (and therefore possibly contains some inactive warps).
	if (blockIdx.x == gridDim.x - 1) {
		// If it does, first calculate the amount of active warps in this last active block.
		int activeWarpsTotal = (N % warpSize == 0) ? (N / warpSize) : (N / warpSize + 1);
		int activeWarpsInBlock = activeWarpsTotal % (blockDim.x / warpSize);
		if (activeWarpsInBlock == 0) activeWarpsInBlock = blockDim.x / warpSize;
		
		// Then calculate the first absolute thread ID after the last active
		// thread for the final reduction within the first warp of this block.
		N = activeWarpsInBlock + blockIdx.x * blockDim.x;
	}
	else {
		// If it doesn't, you can directly calculate the first absolute thread ID after the
		// last active thread for the final reduction within the first warp of this block. 
		// (As the amount of active warps in a fully active block is blockDim.x / warpSize).
		N = blockDim.x / warpSize + blockIdx.x * blockDim.x;
	}

	// Final reduce within the first warp of this block
	if (wid == 0) val = warpReduce(val, func, N);

	return val;
}