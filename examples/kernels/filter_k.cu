extern  "C" __global__
void filter_k(int* dst, int* nres, const int* src, int n) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i >= n) return;

	if (src[i] % 2 == 1){
	    int j = atomicAdd(nres, 1);
		dst[j] = src[i];
		printf("Test %i, %i\n", j, *nres);
		printf("Tes  %i, %i\n", i, src[i]);
	}
}