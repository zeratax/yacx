template<int N, typename T>
extern "C" __global__ void my_kernel(T* array, T data) {
     for( int i=0; i<N; ++i ) {
         array[i] = data;
     }
 }

