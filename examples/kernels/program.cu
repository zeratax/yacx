// template<int N, typename T>
// __global__ void my_kernel(T* data) {
//     T data0 = data[0];
//     for( int i=0; i<N-1; ++i ) {
//         data[0] *= data0;
//     }
// }
template<typename type, int size>
__global__ void setKernel(type[] c, type val) {
    auto idx = threadIdx.x * size;

    #pragma unroll(size)
    for (auto i = 0; i < size; i++) {
        c[idx] = val;
        idx++;
    }
}
