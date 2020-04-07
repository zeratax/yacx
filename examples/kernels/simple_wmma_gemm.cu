/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <mma.h>

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Performs an MxNxK GEMM (C=alpha*A*B + beta*C) assuming:
//  1) Matrices are packed in memory.
//  2) M, N and K are multiples of 16. 
//  3) B is transposed, A isn't.
extern "C" __global__
void simple_wmma_gemm(__half* a, __half* b, float* c, float* d, int m_ld, int n_ld, int k_ld, float alpha, float beta)
{
	// Leading dimensions. B is transposed.
	int lda = k_ld;
	int ldb = k_ld;
	int ldc = n_ld;

	// Declare the fragments
	wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> b_frag;
	wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
	wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

	// Tile using a 2D grid

	// Check if there are any tiles left in Y-direction
	for (int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
		warpN < n_ld / 16;
		warpN += gridDim.y * blockDim.y) {
		
		// Check if there are any tiles left in X-direction
		for (int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
			warpM < m_ld / 16;
			warpM += gridDim.x * (blockDim.x / 32)) {

			wmma::fill_fragment(acc_frag, 0.0f);

			// Loop over k
			for (int i = 0; i < k_ld; i += WMMA_K) {
				int aCol = i;
				int aRow = warpM * WMMA_M;

				int bCol = i;
				int bRow = warpN * WMMA_N;

				// Bounds checking
				if (aRow < m_ld && aCol < k_ld && bRow < n_ld && bCol < k_ld) {
					// Load the inputs
					wmma::load_matrix_sync(a_frag, a + aCol + aRow * lda, lda);
					wmma::load_matrix_sync(b_frag, b + bCol + bRow * ldb, ldb);

					// Perform the matrix multiplication
					wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

				}
			}

			// Load in the current value of c, scale it by beta, and add this our result scaled by alpha
			int cCol = warpN * WMMA_N;
			int cRow = warpM * WMMA_M;

			if (cRow < m_ld && cCol < n_ld) {
				wmma::load_matrix_sync(c_frag, c + cCol + cRow * ldc, ldc, wmma::mem_row_major);

				for (int i = 0; i < c_frag.num_elements; i++) {
					c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
				}

				// Store the output
				wmma::store_matrix_sync(d + cCol + cRow * ldc, c_frag, ldc, wmma::mem_row_major);
			}

		}
	}

}
