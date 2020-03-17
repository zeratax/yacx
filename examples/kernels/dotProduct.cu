extern "C" __global__
void saxpy(float *x, float *y, float *out) {
   out[threadIdx.x] = a * x[threadIdx.x] + y[threadIdx.x];
}
