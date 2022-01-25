import java.io.IOException;

import yacx.Executor;

public class ExampleSimpleGEMMBenchmark {

	public static void main(String[] args) throws IOException {
		final int WMMA_M = 16;
		final int WMMA_N = 16;

		// Load library
		Executor.loadLibrary();

		// Benchmark Simple-GEMM-Kernel
		new MatrixUtils.BenchmarkGEMM() {

			@Override
			public int getGrid0(MatrixUtils.MatrixDimensions dims) {
				int m = (dims.m % WMMA_M == 0) ? dims.m : (dims.m / WMMA_M + 1) * WMMA_M;
				int blockDimX = getBlock0(dims);
				return (m + (WMMA_M * blockDimX / 32 - 1)) / (WMMA_M * blockDimX / 32);
			}

			@Override
			public int getGrid1(MatrixUtils.MatrixDimensions dims) {
				int blockDimY = getBlock1(dims);
				return (dims.n + WMMA_N * blockDimY - 1) / (WMMA_N * blockDimY);
			}

			@Override
			public int getBlock0(MatrixUtils.MatrixDimensions dims) {
				return 128;
			}

			@Override
			public int getBlock1(MatrixUtils.MatrixDimensions dims) {
				return 4;
			}

			@Override
			public MatrixUtils.MatrixDimensions getPaddingDim(MatrixUtils.MatrixDimensions dims) {
				return new MatrixUtils.MatrixDimensions((dims.m % 16) == 0 ? dims.m : (dims.m / 16 + 1) * 16,
						(dims.n % 16) == 0 ? dims.n : (dims.n / 16 + 1) * 16, (dims.k % 16) == 0 ? dims.k : (dims.k / 16 + 1) * 16);
			}
		}.benchmark("simple_wmma_gemm");
	}
}