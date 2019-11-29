template <typename type, int size>
__global__ void my_kernel(type[] c, type val) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;

#pragma unroll(size)
  for (auto i = idx * size; i < idx * size + size; i++) {
    c[i] = idx + val;
  }
}