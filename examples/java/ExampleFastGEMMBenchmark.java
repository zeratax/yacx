import java.io.IOException;

import yacx.Device;
import yacx.Executor;

public class ExampleFastGEMMBenchmark {

	public static void main(String[] args) throws IOException {
		// Load Libary
		Executor.loadLibary();

		// Benchmark Simple-GEMM-Kernel
		System.out.println(new MatrixUtils.BenchmarkGEMM() {

			@Override
			public int getGrid0(int dim) {
				return 32 * 8;
			}

			@Override
			public int getBlock0(int dim) {
				return Device.createDevice().getMultiprocessorCount();
			}
		}.benchmark("fast_wmma_gemm"));
	}
}
