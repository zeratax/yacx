import java.io.IOException;

import yacx.ArrayArg;
import yacx.Executor;
import yacx.Executor.BenchmarkResult;
import yacx.FloatArg;
import yacx.HalfArg;
import yacx.IntArg;
import yacx.KernelArg;
import yacx.Options;

public class MatrixUtils {
	public final static long KB = 1024;
	public final static long MB = 1024 * 1024;

	/**
	 * Prints a matrix.
	 * 
	 * @param matrix  matrix, which should be printed
	 * @param columns number of columns of the matrix
	 */
	public static void printlnMatrix(float[] matrix, int columns) {
		assert (matrix.length % columns == 0);

		int rows = matrix.length / columns;

		int stringLengthElement = 3;
		for (int i = 0; i < matrix.length; i++)
			if (("" + matrix[i]).length() > stringLengthElement)
				stringLengthElement = ("" + matrix[i]).length();

		for (int row = 0; row < rows; row++) {
			for (int column = 0; column < columns; column++) {
				String elementString = "" + matrix[row * columns + column];

				for (int i = elementString.length(); i < stringLengthElement + 1; i++)
					System.out.print(" ");

				System.out.print(elementString);
			}

			System.out.println();
		}
		System.out.println();
	}

	public static abstract class BenchmarkGEMM extends Executor.KernelArgCreator {
		private final long[] dataSizes = new long[] { 1 * KB, 4 * KB, 16 * KB, 64 * KB, 256 * KB, 1 * MB, 4 * MB,
				16 * MB, 64 * MB, 256 * MB, 512 * MB, 1024 * MB };

		@Override
		public int getDataLength(long dataSizeBytes) {
			return (int) Math.sqrt(dataSizeBytes / FloatArg.SIZE_BYTES);
		}

		public KernelArg[] createMatrixPadding(ArrayArg aMatrixArg, ArrayArg bMatrixArg, ArrayArg cMatrixArg,
				ArrayArg dMatrixArg, int dim) {
			return new KernelArg[] { aMatrixArg, bMatrixArg, cMatrixArg, dMatrixArg };
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
			HalfArg bMatrixArg = HalfArg.createTransposed(bMatrix, dim, dim);
			FloatArg cMatrixArg = FloatArg.create(cMatrix);
			FloatArg dMatrixArg = FloatArg.createOutput(dim * dim);
			KernelArg mArg = IntArg.createValue(dim);
			KernelArg nArg = IntArg.createValue(dim);
			KernelArg kArg = IntArg.createValue(dim);
			KernelArg alphaArg = FloatArg.createValue(alpha);
			KernelArg betaArg = FloatArg.createValue(beta);

			KernelArg[] kernelArgsPadding = createMatrixPadding(aMatrixArg, bMatrixArg, cMatrixArg, dMatrixArg, dim);

			return new KernelArg[] { kernelArgsPadding[0], kernelArgsPadding[1], kernelArgsPadding[2],
					kernelArgsPadding[3], mArg, nArg, kArg, alphaArg, betaArg };
		}

		public void benchmark(String kernel) throws IOException {
			Options options = Options.createOptions("--gpu-architecture=compute_70");

			// Warm up
			Executor.benchmark(kernel, options, 30, this, 256 * MB);

			BenchmarkResult result = Executor.benchmark(kernel, options, 50, this, dataSizes);

			String resultString = result.toString();
			for (long dataSize : dataSizes) {
				resultString = resultString.replaceFirst("B: execution-time:", "B (" + getDataLength(dataSize) + "x"
						+ getDataLength(dataSize) + " matrices): execution-time:");
			}

			System.out.println(resultString);
		}
	}
}
