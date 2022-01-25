import java.io.IOException;

import yacx.Executor;
import yacx.Executor.BenchmarkResult;
import yacx.FloatArg;
import yacx.HalfArg;
import yacx.IntArg;
import yacx.KernelArg;
import yacx.Options;
import yacx.PaddingArg;

public class MatrixUtils {
	private final static String compilerArg = "--gpu-architecture=compute_70";

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
	
	public static class MatrixDimensions {
		public final int m, n, k;
		
		public MatrixDimensions(int dim) {
			this.m = dim;
			this.n = dim;
			this.k = dim;
		}
		
		public MatrixDimensions(int m, int n, int k) {
			this.m = m;
			this.n = n;
			this.k = k;
		}
		
		@Override
		public String toString() {
			return "(m = " + m + ", n = " + n + ", k = " + k + ")";
		}
		
		@Override
		public boolean equals(Object o) {
			if (o == null || !(o instanceof MatrixDimensions))
				return false;
			
			MatrixDimensions md = (MatrixDimensions) o;
			return  md.m == m && md.n == n && md.k == k;
		}
	}

	private final static MatrixDimensions[] dataSizes = new MatrixDimensions[] { new MatrixDimensions(256), new MatrixDimensions(512),
			new MatrixDimensions(1024), new MatrixDimensions(2048)};
	
	public static abstract class BenchmarkGEMM extends Executor.KernelArgCreator<MatrixDimensions> {
		public MatrixDimensions getPaddingDim(MatrixDimensions dim) {
			return dim;
		}

		@Override
		public KernelArg[] createArgs(MatrixDimensions dims) {
			float alpha = 1f;
			float beta = 1f;
			float[] aMatrix = new float[dims.m * dims.k];
			float[] bMatrix = new float[dims.k * dims.n];
			float[] cMatrix = new float[dims.m * dims.n];
			for (int i = 0; i < aMatrix.length; i++) {
				aMatrix[i] = i;
			}
			for (int i = 0; i < bMatrix.length; i++) {
				bMatrix[i] = i;
			}
			for (int i = 0; i < cMatrix.length; i++) {
				cMatrix[i] = i;
			}
			
			MatrixDimensions paddingDim = getPaddingDim(dims);

			HalfArg aMatrixArg = HalfArg.create(aMatrix);
			HalfArg bMatrixArg = HalfArg.createTransposed(bMatrix, dims.k, dims.n);
			FloatArg cMatrixArg = FloatArg.create(cMatrix);
			FloatArg dMatrixArg = FloatArg.createOutput(dims.m * dims.n);
			KernelArg mArg = IntArg.createValue(paddingDim.m);
			KernelArg nArg = IntArg.createValue(paddingDim.n);
			KernelArg kArg = IntArg.createValue(paddingDim.k);
			KernelArg alphaArg = FloatArg.createValue(alpha);
			KernelArg betaArg = FloatArg.createValue(beta);

			KernelArg aMatrixArgPadding;
			KernelArg bMatrixArgPadding;
			KernelArg cMatrixArgPadding;
			KernelArg dMatrixArgPadding;

			if (!dims.equals(paddingDim)) {
				aMatrixArgPadding = PaddingArg.createMatrixPadding(aMatrixArg, dims.m, dims.k, paddingDim.m, paddingDim.k, 0);
				bMatrixArgPadding = PaddingArg.createMatrixPadding(bMatrixArg, dims.n, dims.k, paddingDim.n, paddingDim.k, 0);
				cMatrixArgPadding = PaddingArg.createMatrixPadding(cMatrixArg, dims.m, dims.n, paddingDim.m, paddingDim.n, 0);
				dMatrixArgPadding = PaddingArg.createMatrixPadding(dMatrixArg, dims.m, dims.n, paddingDim.m, paddingDim.n, 0);
			} else {
				aMatrixArgPadding = aMatrixArg;
				bMatrixArgPadding = bMatrixArg;
				cMatrixArgPadding = cMatrixArg;
				dMatrixArgPadding = dMatrixArg;
			}

			return new KernelArg[] { aMatrixArgPadding, bMatrixArgPadding, cMatrixArgPadding, dMatrixArgPadding, mArg,
					nArg, kArg, alphaArg, betaArg };
		}

		public void benchmark(String kernel) throws IOException {
			Options options = Options.createOptions(compilerArg);

			// Test dataSize with and without Padding
			MatrixDimensions[] dataSizes = new MatrixDimensions[MatrixUtils.dataSizes.length * 2];
			for (int i = 0; i < dataSizes.length; i += 2) {
				dataSizes[i+1] = MatrixUtils.dataSizes[i / 2];
				dataSizes[i] = new MatrixDimensions(dataSizes[i+1].m + 1, dataSizes[i+1].n + 1, dataSizes[i+1].k + 1);
			}

			// Warm up
			Executor.benchmark(kernel, options, 30, this, new MatrixDimensions(2048));

			BenchmarkResult<MatrixDimensions> result = Executor.benchmark(kernel, options, 50, this, dataSizes);

			System.out.println(result);
		}
	}
}
