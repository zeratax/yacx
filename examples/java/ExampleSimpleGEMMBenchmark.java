import java.io.IOException;

import yacx.ArrayArg;
import yacx.Executor;
import yacx.KernelArg;
import yacx.PaddingArg;

public class ExampleSimpleGEMMBenchmark {

	public static void main(String[] args) throws IOException {
		final int WMMA_M = 16;
		final int WMMA_N = 16;

		// Load Libary
		Executor.loadLibary();

		// Benchmark Simple-GEMM-Kernel
		new MatrixUtils.BenchmarkGEMM() {

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

			@Override
			public KernelArg[] createMatrixPadding(ArrayArg aMatrixArg, ArrayArg bMatrixArg, ArrayArg cMatrixArg,
					ArrayArg dMatrixArg, int dim) {
				if (dim % 16 != 0) {
					int dimPadding = (dim / 16 + 1) * 16;

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
		}.benchmark("simple_wmma_gemm");
	}
}