import java.io.IOException;

import yacx.ArrayArg;
import yacx.Device;
import yacx.Executor;
import yacx.FloatArg;
import yacx.HalfArg;
import yacx.KernelArg;
import yacx.PaddingArg;

public class ExampleFastGEMMBenchmark {

	public static void main(String[] args) throws IOException {
		// Constants for shared memory calculation
		final int M = 16;
		final int N = 16;
		final int K = 16;
		// If you change this, don't forget to adjust the SHARED_MEMORY_LIMIT_64K in the
		// kernel, too.
		final boolean SHARED_MEMORY_LIMIT_64K = false;
		final int BLOCK_ROW_WARPS = 2;
		final int BLOCK_COL_WARPS = 4;
		final int WARP_ROW_TILES = 4;
		final int WARP_COL_TILES = 2;
		final int BLOCK_COL_TILES = WARP_COL_TILES * BLOCK_COL_WARPS;
		final int CHUNK_K = SHARED_MEMORY_LIMIT_64K ? 4 : 8;
		final int SKEW_HALF = 8;

		// Load Libary
		Executor.loadLibary();

		// Compute required shared memory size
		final long SHMEM_SZ = Math.max(HalfArg.SIZE_BYTES * (BLOCK_COL_TILES * M) * (CHUNK_K * K + SKEW_HALF) * 2,
				M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * N * (BLOCK_COL_WARPS * WARP_COL_TILES) * FloatArg.SIZE_BYTES);

		// Calculate and print out the required and available shared memory size
		long required = SHMEM_SZ / 1024;
		long available = Device.createDevice().getSharedMemPerMultiprocessor() / 1024;
		System.out.println("Required shared memory size per multiprocessor: " + required + " KB");
		System.out.println("Available shared memory size per multiprocessor: " + available + " KB");

		// Check if there's enough shared memory per block available on the device for
		// this kernel
		if (required > available) {
			System.out.println("Not enough shared memory per block available on the device for this kernel! Abort!");
			System.out.println(
					"Please use the simple GEMM kernel instead or increase the amount of shared memory per block if possible!");
			System.exit(1);
		}

		// Benchmark Simple-GEMM-Kernel
		new MatrixUtils.BenchmarkGEMM() {

			@Override
			public int getGrid0(int dim) {
				return Device.createDevice().getMultiprocessorCount();
			}

			@Override
			public int getBlock0(int dim) {
				return 32 * 8;
			}

			@Override
			public long getSharedMemory(long dataSizeBytes) {
				return SHMEM_SZ;
			}

			@Override
			public KernelArg[] createMatrixPadding(ArrayArg aMatrixArg, ArrayArg bMatrixArg, ArrayArg cMatrixArg,
					ArrayArg dMatrixArg, int dim) {
				if (dim % 128 != 0) {
					int dimPadding = (dim / 128 + 1) * 128;

					KernelArg aMatrixArgPadding = PaddingArg.createMatrixPadding(aMatrixArg, dim, dim, dimPadding,
							dimPadding, 0);
					KernelArg bMatrixArgPadding = PaddingArg.createMatrixPadding(bMatrixArg, dim, dim, dimPadding,
							dimPadding, 0);
					KernelArg cMatrixArgPadding = PaddingArg.createMatrixPadding(cMatrixArg, dim, dim, dimPadding,
							dimPadding, 0);
					KernelArg dMatrixArgPadding = PaddingArg.createMatrixPadding(dMatrixArg, dim, dim, dimPadding,
							dimPadding, 0);

					return new KernelArg[] { aMatrixArgPadding, bMatrixArgPadding, cMatrixArgPadding,
							dMatrixArgPadding };
				} else {
					return super.createMatrixPadding(aMatrixArg, bMatrixArg, cMatrixArg, dMatrixArg, dim);
				}
			}
		}.benchmark("fast_wmma_gemm");
	}
}
