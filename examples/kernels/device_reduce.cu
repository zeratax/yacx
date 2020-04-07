#include "kernels/block_reduce.h"

// Reduction function, currently the minimum of a and b
// Common alternatives would be:
// 	- sum 	 	(return a + b;)
// 	- product	(return a * b;)
//  - maximum 	(return max(a,b);)
__device__ long long func(long long a, long long b) {
	return a+b;
}

// Performs a device-wide reduction with func as the reduction function while ensuring that inactive threads, 
// whose shuffle value is always zero, are being ignored during the reduction as these might distort the
// result for certain reduction functions (e.g. minimum or multiplication).
// N has to be equal to the amount of elements in the input array.
// Also note that this kernel has to be called twice to work correctly:
// Once to compute the result of each block and once to reduce the results of all blocks to one final result.
extern "C" __global__ 
void device_reduce(long long* in, long long* out, int N) {
	
	// read the value of the input array corresponding with the absolute thread ID
	long long result = in[blockIdx.x * blockDim.x + threadIdx.x];

	// Performs the reduction function manually for some values if the amount of
	// active threads is smaller than the amount of elements in the input array.
	for (int i = blockIdx.x * blockDim.x + threadIdx.x + blockDim.x * gridDim.x; i < N; i += blockDim.x * gridDim.x) {
		result = func(result, in[i]);
	}

	// Each block performs the reduction
	result = blockReduce(result, &func, N);
	
	// Saves the result of the first thread in each block, which
	// is equal to the result of the block-wide reduction,
	// in the corresponding position in the output array.
	if (threadIdx.x == 0) out[blockIdx.x] = result;
}
