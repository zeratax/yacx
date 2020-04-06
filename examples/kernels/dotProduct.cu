extern "C" __global__
void dotProduct(float* vecA, float* vecB, float* output){
  atomicAdd(output, vecA[threadIdx.x] * vecB[threadIdx.x]);
}
