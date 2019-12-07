
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

// filter for uneven numbers with atomic add
__global__ void filter_k(int* dst, int* nres, const int* src, int n) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	// boundary check
	if (i >= n) return;

	// saves the element to its calculated position in the destination array if it satisfies the predicate
	if (src[i] % 2 == 1)
		dst[atomicAdd(nres, 1)] = src[i];
}

int main() {
	// set array size
	const int arraySize = 32768;

	// generate the input array on the host
	int h_in[arraySize];
	for (int i = 0; i < arraySize; i++) {
		h_in[i] = i;
	}

	const int arrayBytes = arraySize * sizeof(int);

	// generate the output on the host
	int *h_out = (int*)malloc(arrayBytes);
	int x = 0;
	int* h_counter = &x;

	// declare the GPU memory pointer
	int* d_in;
	int* d_out;
	int* d_counter;

	// allocate GPU memory
	cudaMalloc((void**)&d_in, arrayBytes);
	cudaMalloc((void**)&d_out, arrayBytes);
	cudaMalloc((void**)&d_counter, sizeof(int));

	// transfer the array to the GPU
	cudaMemcpy(d_in, h_in, arrayBytes, cudaMemcpyHostToDevice);

	// calculate amounts of blocks and threads
	int blocks = arraySize / 1024 + 1;
	int threads = (blocks > 1) ? 1024 : arraySize;

	// launch the kernel
	filter_k<<<blocks, threads>>>(d_out, d_counter, d_in, arraySize);

	// copy back the results to the CPU
	cudaMemcpy(h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_out, d_out, *h_counter * sizeof(int), cudaMemcpyDeviceToHost);

	// print out the results
	printf("Counter = %d\n", *h_counter);

	// free the GPU memory allocation
	cudaFree(d_in);
	cudaFree(d_out);
	cudaFree(d_counter);

	return 0;
}