import java.io.IOException;
import java.util.Arrays;

import yacx.Executor;
import yacx.FloatArg;
import yacx.HalfArg;
import yacx.IntArg;
import yacx.KernelArg;
import yacx.Options;
import yacx.PaddingArg;
import yacx.Utils;

public class ExampleSimpleGEMMBenchmark {
	private final static int KB = 1024;
	private final static int MB = 1024 * 1024;

	public static void main(String[] args) throws IOException {
		// Load Libary
		Executor.loadLibary();

		// Benchmark Simple-GEMM-Kernel
		System.out.println(Executor.benchmark("simple_wmma_gemm", Options.createOptions("--gpu-architecture=compute_70"), 10, new Executor.KernelArgCreator() {

			@Override
			public int getDataLength(int dataSizeBytes) {
				return (int) (dataSizeBytes / FloatArg.SIZE_BYTES);
			}

			public int getDimension(int dataLength) {
				return (int) Math.sqrt(dataLength);
			}

			@Override
			public int getGrid0(int dataLength) {
				int x = getDimension(dataLength);
				int m = (x % 16 == 0) ? x : (x / 16 + 1) * 16;
				int WMMA_M = 16;
				int blockDimX = getBlock0(0);
				return (m + (WMMA_M * blockDimX / 32 - 1)) / (WMMA_M * blockDimX / 32);
			}
			
			@Override
			public int getGrid1(int dataLength) {
				int z = getDimension(dataLength);
				int n = (z % 16 == 0) ? z : (z / 16 + 1) * 16;
				int WMMA_N = 16;
				int blockDimY = getBlock1(0);
				return (n + WMMA_N * blockDimY - 1) / (WMMA_N * blockDimY);
			}

			@Override
			public int getBlock0(int dataLength) {
				return 128;
			}
			
			@Override
			public int getBlock1(int dataLength) {
				return 4;
			}

			@Override
			public KernelArg[] createArgs(int dataLength) {
				int x = getDimension(dataLength);
				int m = (x % 16 == 0) ? x : (x / 16 + 1) * 16;
				float alpha = 1f;
				float beta = 1f;
				float[] aMatrix = new float[dataLength];
				float[] bMatrix = new float[dataLength];
				float[] cMatrix = new float[dataLength];
				for (int i = 0; i < aMatrix.length; i++) {
					aMatrix[i] = i;
				}
				for (int i = 0; i < bMatrix.length; i++) {
					bMatrix[i] = i;
				}
				for (int i = 0; i < cMatrix.length; i++) {
					cMatrix[i] = i;
				}
				
				HalfArg aMatrixArg = HalfArg.create(aMatrix);
				HalfArg bMatrixArg = HalfArg.createTransposed(x, x, bMatrix);
				FloatArg cMatrixArg = FloatArg.create(cMatrix);
				FloatArg dMatrixArg = FloatArg.createOutput(x * x);
				KernelArg mArg = IntArg.createValue(m);
				KernelArg nArg = IntArg.createValue(m);
				KernelArg kArg = IntArg.createValue(m);
				KernelArg alphaArg = FloatArg.createValue(alpha);
				KernelArg betaArg = FloatArg.createValue(beta);
				
				PaddingArg aMatrixArgPadding = PaddingArg.createMatrixPadding(aMatrixArg, x, x, m, m, 0);
				PaddingArg bMatrixArgPadding = PaddingArg.createMatrixPadding(bMatrixArg, x, x, m, m, 0);
				PaddingArg cMatrixArgPadding = PaddingArg.createMatrixPadding(cMatrixArg, x, x, m, m, 0);
				PaddingArg dMatrixArgPadding = PaddingArg.createMatrixPadding(dMatrixArg, x, x, m, m, 0);

				return new KernelArg[] { aMatrixArgPadding, bMatrixArgPadding, cMatrixArgPadding, dMatrixArgPadding,
										 mArg, nArg, kArg, alphaArg, betaArg };
			}
		}, 1 * KB, 4 * KB, 16 * KB, 64 * KB, 256 * KB, 1 * MB, 4 * MB, 16 * MB, 64 * MB, 256 * MB));
	}
}