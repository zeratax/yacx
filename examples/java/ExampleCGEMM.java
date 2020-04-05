import java.io.IOException;

import yacx.Executor;
import yacx.FloatArg;
import yacx.IntArg;
import yacx.KernelArg;
import yacx.Options;
import yacx.Utils;

public class ExampleCGEMM {
	public static void main(String[] arguments) throws IOException {
		// Load Library
		Executor.loadLibrary();

		// Load Kernel from file
		String cFunctionName = "gemm";
		String cProgram = Utils.loadFile("kernels/" + cFunctionName + ".c");

		// Dimensions of matrix
		int x = 4;
		int y = 3;
		int z = 2;

		// Test data
		float alpha = 1f;
		float beta = 1f;
		float[] aMatrix = new float[x * y];
		float[] bMatrix = new float[y * z];
		float[] cMatrix = new float[x * z];
		for (int i = 0; i < aMatrix.length; i++) {
			aMatrix[i] = i + 1;
		}
		for (int i = 0; i < bMatrix.length; i++) {
			bMatrix[i] = x * y + i + 1;
		}
		for (int i = 0; i < cMatrix.length; i++) {
			cMatrix[i] = 2 * (i + 1);
		}

		// KernelArgs
		FloatArg aMatrixArg = FloatArg.create(aMatrix);
		FloatArg bMatrixArg = FloatArg.create(bMatrix);
		FloatArg cMatrixArg = FloatArg.create(cMatrix);
		FloatArg dMatrixArg = FloatArg.createOutput(x * z);
		KernelArg mArg = IntArg.createValue(x);
		KernelArg nArg = IntArg.createValue(y);
		KernelArg kArg = IntArg.createValue(z);
		KernelArg alphaArg = FloatArg.createValue(alpha);
		KernelArg betaArg = FloatArg.createValue(beta);

		KernelArg[] args = new KernelArg[] { dMatrixArg, mArg, nArg, kArg, aMatrixArg, bMatrixArg, cMatrixArg, alphaArg,
				betaArg };

		// Optional compiler and compiler arguments
		String compiler = "gcc";
		Options options = Options.createOptions("-O3");

		Executor.executeC(cProgram, cFunctionName, compiler, options, args);

		// Print Result
		System.out.println("CProgram gemm launched");
		System.out.println();
		System.out.println("aMatrix:");
		MatrixUtils.printlnMatrix(aMatrix, y);
		System.out.println("bMatrix:");
		MatrixUtils.printlnMatrix(bMatrix, z);
		System.out.println("cMatrix:");
		MatrixUtils.printlnMatrix(cMatrix, z);
		System.out.println("resultmatrix:");
		MatrixUtils.printlnMatrix(dMatrixArg.asFloatArray(), z);
	}
}
