template<typename T>
__global__ void f3(int *result) {
	*result = sizeof(T);
}