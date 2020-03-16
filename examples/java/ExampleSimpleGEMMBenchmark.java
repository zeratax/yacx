import java.io.IOException;

import yacx.Executor;

public class ExampleSimpleGEMMBenchmark {

	public static void main(String[] args) throws IOException {
		// Load Libary
		Executor.loadLibary();

		// Benchmark Simple-GEMM-Kernel
		System.out.println(new MatrixUtils.BenchmarkGEMM() {

			@Override
			public int getGrid0(int dim) {
				int m = (dim % 16 == 0) ? dim : (dim / 16 + 1) * 16;
				int WMMA_M = 16;
				int blockDimX = getBlock0(0);
				return (m + (WMMA_M * blockDimX / 32 - 1)) / (WMMA_M * blockDimX / 32);
			}

			@Override
			public int getGrid1(int dim) {
				int WMMA_N = 16;
				int blockDimY = getBlock1(0);
				return (dim + WMMA_N * blockDimY - 1) / (WMMA_N * blockDimY);
			}

			@Override
			public int getBlock0(int dim) {
				return 128;
			}

			@Override
			public int getBlock1(int dim) {
				return 4;
			}
		}.benchmark("simple_wmma_gemm"));
	}
}