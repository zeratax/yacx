import java.io.IOException;
import java.util.Arrays;

import yacx.Executor;
import yacx.Executor.BenchmarkResult;
import yacx.FloatArg;
import yacx.HalfArg;
import yacx.IntArg;
import yacx.KernelArg;
import yacx.Options;

public class MatrixUtils {
	public final static int KB = 1024;
	public final static int MB = 1024 * 1024;

	// TODO Die Methoden sind total langsam
	public static void printlnMatrix(FloatArg floatMatrix, int columns) {
		int rows = floatMatrix.getLength() / columns;

		for (int r = 0; r < rows - 1; r++) {
			System.out.println(Arrays.toString(floatMatrix.slice(r * columns, (r + 1) * columns - 1).asFloatArray()));
		}
		System.out.println(Arrays.toString(floatMatrix.slice((rows - 1) * columns, rows * columns - 1).asFloatArray()));
	}

	public static void printlnMatrix(HalfArg halfMatrix, int columns) {
		int rows = halfMatrix.getLength() / columns;

		for (int r = 0; r < rows - 1; r++) {
			System.out.println(Arrays.toString(halfMatrix.slice(r * columns, (r + 1) * columns - 1).asFloatArray()));
		}
		System.out.println(Arrays.toString(halfMatrix.slice((rows - 1) * columns, rows * columns - 1).asFloatArray()));
	}

	public static abstract class BenchmarkGEMM extends Executor.KernelArgCreator {
		@Override
		public int getDataLength(int dataSizeBytes) {
			return (int) Math.sqrt(dataSizeBytes / FloatArg.SIZE_BYTES);
		}

		@Override
		public KernelArg[] createArgs(int dim) {
			float alpha = 1f;
			float beta = 1f;
			float[] aMatrix = new float[dim * dim];
			float[] bMatrix = new float[dim * dim];
			float[] cMatrix = new float[dim * dim];
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
			HalfArg bMatrixArg = HalfArg.createTransposed(dim, dim, bMatrix);
			FloatArg cMatrixArg = FloatArg.create(cMatrix);
			FloatArg dMatrixArg = FloatArg.createOutput(dim * dim);
			KernelArg mArg = IntArg.createValue(dim);
			KernelArg nArg = IntArg.createValue(dim);
			KernelArg kArg = IntArg.createValue(dim);
			KernelArg alphaArg = FloatArg.createValue(alpha);
			KernelArg betaArg = FloatArg.createValue(beta);

			return new KernelArg[] { aMatrixArg, bMatrixArg, cMatrixArg, dMatrixArg, mArg, nArg, kArg, alphaArg,
					betaArg };
		}

		public BenchmarkResult benchmark(String kernel) throws IOException {
			Options options = Options.createOptions("--gpu-architecture=compute_70");

			//Warm up
			Executor.benchmark(kernel, options, 10, this, 256 * MB);
			
			return Executor.benchmark(kernel, options, 20, this, 1 * KB, 4 * KB, 16 * KB, 64 * KB, 256 * KB, 1 * MB,
					4 * MB, 16 * MB, 64 * MB, 256 * MB);
		}
	}
}
