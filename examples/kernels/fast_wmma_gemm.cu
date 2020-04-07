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

// In this program, the compute_gemm kernel computes the result of a matrix
// multiplication and addition: D = alpha * A * B + beta * C. The dimensions of
// both C and D matrices are m_global x n_global. The A matrix is m_global x
// k_global (row-major), the B matrix is k_global x n_global (column-major). In
// that kernel, each CTA computes one 128 x 128 tile of the resulting matrix per
// iteration. When the tile is computed, the CTA stores it to the global memory
// and begins a new iteration, selecting a new 128 x 128 tile to compute.
// Each CTA consists of eight warps. For the 128 x 128 tile, each warp computes
// eight 16 x 16 subtiles, organized in a 2 x 4 two-dimensional array. Warps
// compute the 16 x 16 subtiles using nvcuda::wmma::mma_sync operations by
// moving through the k_global dimension of the A and B matrices and
// accumulating the intermediate result in the local thread state.

// There are a number of simple optimizations used in the algorithm:
// - The CTA copies the 128 x 128 tile of the C matrix from the global memory to
//   shared memory. After that is done, each warp loads the C matrix fragments
//   from shared memory, thus avoiding a random global memory access.
// - On each internal iteration, the CTA copies a portion of the A and B
// matrices from
//   global memory to shared memory. After that, all warps in the CTA reuse the
//   A and B data from shared memory, thus reducing the number of data copies
//   from global memory.
// - The portions of the A and B matrices are stored in shared memory with an
// additional
//   padding (skew) to reduce the number of shared memory access bank conflicts.
//   (See a detailed explanation near the SKEW_HALF macro definition.)
// - When the CTA finishes computing the tiles of the resulting matrix, each
// warp stores
//   its subtiles to shared memory. The CTA then copies the shared memory
//   contents to global memory, again avoiding redundant random global memory
//   accesses.
// - Note that the CTA tile size is chosen to maximize the GPU register
// utilization,
//   but carefully enough to avoid local memory use.

// Note: The dimension of the input matrices (m_global, n_global & k_global) have to be multiples of 128 for the kernel to work correctly.

#include <mma.h>

using namespace nvcuda;

#ifndef SHARED_MEMORY_LIMIT_64K
// Set this to 0 to use more than 64 Kb of shared memory to cache data, to
// improve the performance of the computations on GPU.
// Note that you need a GPU that can have more than 64 Kb of shared memory
// per multiprocessor.
#define SHARED_MEMORY_LIMIT_64K 1
#endif

// GPU configuration.

#define WARP_SIZE 32

// MMA matrix tile dimensions.
#define M 16
#define N 16
#define K 16

// GEMM configuration.

#define X_GLOBAL 2
#define Y_GLOBAL 2
#define Z_GLOBAL 17

#define TEMP ((X_GLOBAL > Y_GLOBAL) ? X_GLOBAL : Y_GLOBAL)
#define MAX_DIMENSION ((TEMP > Z_GLOBAL) ? TEMP : Z_GLOBAL)

#define M_GLOBAL ((MAX_DIMENSION % 16 == 0) ? MAX_DIMENSION : ((MAX_DIMENSION / 16 + 1) * 16))
#define N_GLOBAL M_GLOBAL
#define K_GLOBAL M_GLOBAL

#define M_TILES (M_GLOBAL / M)
#define N_TILES (N_GLOBAL / N)
#define K_TILES (K_GLOBAL / K)

#define C_LAYOUT wmma::mem_row_major

// Implementation constants.
#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)

#if SHARED_MEMORY_LIMIT_64K
// With only 64 Kb shared memory available, we can fit two 8-tile chunks of
// the A and B matrix data, that are 16 * 16 * 8 * 8 * 2 = 32 Kb each
// (i.e. two 8x8 arrays of tiles of 16x16 half-typed elements per CTA).
// But we cannot account the 8 Kb total skew overhead, without which the
// performance would be severely impacted. So we choose to reduce the chunk size
// in half, i.e. the amount of A and B matrix data we cache in shared memory.
// Accordingly, this doubles the number of outer iterations across the global K
// dimension, which only slightly impacts the performance.
#define CHUNK_K 4
#else
#define CHUNK_K 8
#endif

#define CHUNK_LINE_BYTES (CHUNK_K * K * sizeof(__half))
#define WARP_COPY_BYTES (WARP_SIZE * sizeof(int4))
#define CHUNK_COPY_LINES_PER_WARP (WARP_COPY_BYTES / CHUNK_LINE_BYTES)
#define CHUNK_COPY_LINE_LANES (WARP_SIZE / CHUNK_COPY_LINES_PER_WARP)

#define BLOCK_ROW_WARPS 2
#define BLOCK_COL_WARPS 4

#define WARP_ROW_TILES 4
#define WARP_COL_TILES 2

#define BLOCK_ROW_TILES (WARP_ROW_TILES * BLOCK_ROW_WARPS)
#define BLOCK_COL_TILES (WARP_COL_TILES * BLOCK_COL_WARPS)

#define SHMEM_STRIDE (N * BLOCK_ROW_TILES)
#define SHMEM_OFFSET (N * WARP_ROW_TILES)

// The macro below is used to shift rows of the A matrix and columns of the B
// matrix in shared memory to minimize possible bank conflicts. Before
// performing the nvcuda::wmma::mma_sync operation, the warp must load the
// matrix data using the nvcuda::wmma::load_matrix_sync operation. Although the
// memory access pattern is not specified for that function, each lane in the
// warp can read one or multiple matrix elements from different matrix rows or
// columns. For shared memory, such access can result in bank conflicts if
// different rows / columns of the matrix map to the same bank. By shifting each
// row and column by a few bytes, we make sure that they map to different banks,
// thus reducing the number of possible bank conflicts. The number of 8 two-byte
// "half" elements is chosen as the minimum possible shift because we must keep
// each row and column 128-bit aligned, as required by
// nvcuda::wmma::load_matrix_sync.
#define SKEW_HALF 8

extern "C" __global__
void fast_wmma_gemm(const __half* A, const __half* B, const float* C,
	float* D, float alpha, float beta, int m_global, int n_global, int k_global) {
	extern __shared__ __half shmem[][CHUNK_K * K + SKEW_HALF];
	
	int global_mem_stride = n_global;

	// Warp and lane identification.
	const unsigned int warpId = threadIdx.x / WARP_SIZE;
	const unsigned int laneId = threadIdx.x % WARP_SIZE;

	// Offset in shared memory from which the B matrix is stored.
	const size_t shmem_idx_b_off = BLOCK_COL_TILES * M;

	// This pointer is used to access the C and D matrix tiles this warp computes.
	float* shmem_warp_tile_ptr = (float*)&shmem[0][0] +
		(warpId / 2) * SHMEM_STRIDE * K * 2 +
		(warpId % 2) * SHMEM_OFFSET;

	// This pointer is used to stream the C and D matrices block-wide tile to and
	// from shared memory.
	float* shmem_warp_stream_ptr =
		(float*)&shmem[0][0] + warpId * SHMEM_STRIDE * K;

	// Adjust the beta scaler, as it'll be multiplied by alpha at the end of
	// each tile computation. Technically this is not generally correct (may
	// result in a loss of precision). Zero still needs to be specially handled
	// though.
	beta /= alpha;

	// Each CTA slides along the 128 x 128 tiles from the top left corner of the
	// matrix to the right and down, and selects the next tile to compute. Once
	// there's no such tile, all warps in this CTA exit.
	for (unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
		const unsigned int block_tile_i =
			((block_pos * BLOCK_ROW_TILES) / N_TILES) * (BLOCK_COL_TILES);
		const unsigned int block_tile_j = (block_pos * BLOCK_COL_TILES) % N_TILES;

		// Stop when there are no more D matrix tiles to compute in this CTA.
		if (block_tile_i >= M_TILES) {
			break;
		}

		// This warp's pointer to the C matrix data to copy memory from to shared
		// memory.
		const size_t gmem_idx =
			(block_tile_i + warpId) * M * global_mem_stride + block_tile_j * N;
		const float* src_gmem_warp_stream_ptr = &C[gmem_idx];

		// Stream multiple C tiles to shared memory.
		#pragma unroll
		for (int i = 0; i < K; i++) {
			typedef int4 copy_t;

			*((copy_t*)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId) =
				*((copy_t*)(src_gmem_warp_stream_ptr + global_mem_stride * i) +
					laneId);
		}

		__syncthreads();

		// These fragments will accumulate the result of A and B matrix fragment
		// multiplications along the k_global dimension.
		wmma::fragment<wmma::accumulator, M, N, K, float> c[WARP_COL_TILES]
			[WARP_ROW_TILES];

		// Load the C matrix tiles into fragments from shared memory.
		#pragma unroll
		for (int i = 0; i < WARP_COL_TILES; i++) {
		#pragma unroll
			for (int j = 0; j < WARP_ROW_TILES; j++) {
				const float* tile_ptr =
					shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;

				wmma::load_matrix_sync(c[i][j], tile_ptr, SHMEM_STRIDE, C_LAYOUT);
			}
		}

		__syncthreads();

		// Scale the C matrix.
		#pragma unroll
		for (int i = 0; i < WARP_COL_TILES; i++) {
		#pragma unroll
			for (int j = 0; j < WARP_ROW_TILES; j++) {
		#pragma unroll
				for (int t = 0; t < c[i][j].num_elements; t++) {
					c[i][j].x[t] *= beta;
				}
			}
		}

		// Select what warp copies what matrix to shared memory.
		// Warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.
		const __half* warp_ptr = (warpId < 4) ? (&A[block_tile_i * M * k_global] +
			M * k_global * (warpId % 4) * 2)
			: (&B[block_tile_j * N * k_global] +
				N * k_global * (warpId % 4) * 2);

		// Go through the global K dimension by a fixed step at a time.
		#pragma unroll
		for (int tile_k = 0; tile_k < K_TILES; tile_k += CHUNK_K) {
			// Copy slices of the A and B matrices to shared memory.
			// The first half of the warps in the CTA copy the A matrix, the rest copy
			// the B matrix.
			size_t shmem_idx =
				warpId < (WARPS_PER_BLOCK / 2)
				? (M * (warpId % (WARPS_PER_BLOCK / 2)) * 2)
				: (N * (warpId % (WARPS_PER_BLOCK / 2)) * 2 + shmem_idx_b_off);

			// First half of the warp copies the first row / column of the matrix,
			// the second half of the warp copies the next.
			int4* lane_ptr = (int4*)(warp_ptr + tile_k * K +
				(laneId / CHUNK_COPY_LINE_LANES) * k_global) +
				(laneId % CHUNK_COPY_LINE_LANES);

			// Shift the second half of the warp to the next row / column in the
			// shared memory.
			shmem_idx += laneId / CHUNK_COPY_LINE_LANES;

			#pragma unroll
			for (int i = 0; i < ((WARP_SIZE / 2) / CHUNK_COPY_LINES_PER_WARP) * 2;
				i++) {
				// Copy 16 bytes at once in each lane.
				*((int4*)&shmem[shmem_idx][0] + (laneId % CHUNK_COPY_LINE_LANES)) =
					*lane_ptr;

				// Advance the global memory pointer and the shared memory index.
				lane_ptr =
					(int4*)((__half*)lane_ptr + k_global * CHUNK_COPY_LINES_PER_WARP);
				shmem_idx += CHUNK_COPY_LINES_PER_WARP;
			}

			__syncthreads();

			// Compute a grid of C matrix tiles in each warp.
			#pragma unroll
			for (int k_step = 0; k_step < CHUNK_K; k_step++) {
				wmma::fragment<wmma::matrix_a, M, N, K, __half, wmma::row_major>
					a[WARP_COL_TILES];
				wmma::fragment<wmma::matrix_b, M, N, K, __half, wmma::col_major>
					b[WARP_ROW_TILES];

				#pragma unroll
				for (int i = 0; i < WARP_COL_TILES; i++) {
					size_t shmem_idx_a = (warpId / 2) * M * 2 + (i * M);
					const __half* tile_ptr = &shmem[shmem_idx_a][k_step * K];

					wmma::load_matrix_sync(a[i], tile_ptr, K * CHUNK_K + SKEW_HALF);

					#pragma unroll
					for (int j = 0; j < WARP_ROW_TILES; j++) {
						if (i == 0) {
							// Load the B matrix fragment once, because it is going to be
							// reused against the other A matrix fragments.
							size_t shmem_idx_b = shmem_idx_b_off +
								(WARP_ROW_TILES * N) * (warpId % 2) +
								(j * N);
							const __half* tile_ptr = &shmem[shmem_idx_b][k_step * K];

							wmma::load_matrix_sync(b[j], tile_ptr, K * CHUNK_K + SKEW_HALF);
						}

						wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
					}
				}
			}

			__syncthreads();
		}

		// Store the D fragments to shared memory.
		#pragma unroll
		for (int i = 0; i < WARP_COL_TILES; i++) {
			#pragma unroll
			for (int j = 0; j < WARP_ROW_TILES; j++) {
				#pragma unroll
				// Uniform, point-wise transformations of ALL fragment elements by ALL
				// threads in the warp are well-defined even though element indices
				// within fragment storage are not defined.
				for (int t = 0; t < c[i][j].num_elements; t++) c[i][j].x[t] *= alpha;

				float* tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;

				wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE, C_LAYOUT);
			}
		}

		__syncthreads();

		// Now that shared memory contains all the D tiles, stream them to global
		// memory.
		float* dst_gmem_warp_stream_ptr = &D[gmem_idx];

		#pragma unroll
		for (int i = 0; i < K; i++) {
			*((int4*)(dst_gmem_warp_stream_ptr + global_mem_stride * i) + laneId) =
				*((int4*)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId);
		}

		__syncthreads();
	}
}