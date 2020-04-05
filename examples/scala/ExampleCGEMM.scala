import java.io.IOException;

import yacx.Executor;
import yacx.FloatArg;
import yacx.IntArg;
import yacx.KernelArg;
import yacx.Options;
import yacx.Utils;

object ExampleCGEMM {
	def main(args: Array[String]) : Unit = {
		// Load Library
		Executor.loadLibrary();

		// Load Kernel from file
		val cFunctionName = "gemm";
		val cProgram = Utils.loadFile("kernels/" + cFunctionName + ".c");

		// Dimensions of matrix
		val x = 4;
		val y = 3;
		val z = 2;

		// Test data
		val alpha = 1f;
		val beta = 1f;
		val aMatrix = new Array[Float](x * y);
		val bMatrix = new Array[Float](y * z);
		val cMatrix = new Array[Float](x * z);
		for (i <- 0 until aMatrix.length) {
			aMatrix(i) = i + 1;
		}
		for (i <- 0 until bMatrix.length) {
			bMatrix(i) = x * y + i + 1;
		}
		for (i <- 0 until cMatrix.length) {
			cMatrix(i) = 2 * (i + 1);
		}

		// KernelArgs
		val aMatrixArg = FloatArg.create(aMatrix: _*);
		val bMatrixArg = FloatArg.create(bMatrix: _*);
		val cMatrixArg = FloatArg.create(cMatrix: _*);
		val dMatrixArg = FloatArg.createOutput(x * z);
		val mArg = IntArg.createValue(x);
		val nArg = IntArg.createValue(y);
		val kArg = IntArg.createValue(z);
		val alphaArg = FloatArg.createValue(alpha);
		val betaArg = FloatArg.createValue(beta);

		val args = Array[KernelArg] ( dMatrixArg, mArg, nArg, kArg, aMatrixArg, bMatrixArg, cMatrixArg, alphaArg,
				betaArg );

		// Optional compiler and compiler arguments
		val compiler = "gcc";
		val options = Options.createOptions("-O3");

		Executor.executeC(cProgram, cFunctionName, compiler, options, args: _*);

		// Print Result
		println("CProgram gemm launched");
		println();
		println("aMatrix:");
		MatrixUtils.printlnMatrix(aMatrix, y);
		println("bMatrix:");
		MatrixUtils.printlnMatrix(bMatrix, z);
		println("cMatrix:");
		MatrixUtils.printlnMatrix(cMatrix, z);
		println("resultmatrix:");
		MatrixUtils.printlnMatrix(dMatrixArg.asFloatArray(), z);
	}
}
