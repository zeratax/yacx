import java.io.IOException;

import yacx.Executor;
import yacx.Options;

public class ExampleSimpleGEMMBenchmark {

	public static void main(String[] args) throws IOException {
		final int WMMA_M = 16;
		final int WMMA_N = 16;	
		
		// Load Libary
		Executor.loadLibary();
		
		Options options = Options.createOptions("--gpu-architecture=compute_70");

		// Benchmark Simple-GEMM-Kernel
		System.out.println(new MatrixUtils.BenchmarkGEMM() {

			@Override
			public int getGrid0(int dim) {
				int m = (dim % 16 == 0) ? dim : (dim / 16 + 1) * 16;
				int blockDimX = getBlock0(0);
				return (m + (WMMA_M * blockDimX / 32 - 1)) / (WMMA_M * blockDimX / 32);
			}

			@Override
			public int getGrid1(int dim) {
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