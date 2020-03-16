import java.io.IOException;
import java.util.Arrays;

import yacx.Device;
import yacx.Executor;
import yacx.FloatArg;
import yacx.HalfArg;
import yacx.IntArg;
import yacx.KernelArg;
import yacx.KernelTime;
import yacx.Options;
import yacx.PaddingArg;

public class ExampleFastGEMM {

	public static void main(String[] args) throws IOException {
		// Load Libary
		Executor.loadLibary();

		// Testdata
		int x = 4;
		int y = 3;
		int z = 2;
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

		// Get the next biggest multiple of 128 for each dimension
		int m = (x % 128 == 0) ? x : (x / 128 + 1) * 128;
		int k = (y % 128 == 0) ? y : (y / 128 + 1) * 128;
		int n = (z % 128 == 0) ? z : (z / 128 + 1) * 128;

		// 8 Warps = 256 Threads per Block are required for the kernel to work
		int threads = 32 * 8;
		// The amount of blocks can be freely chosen but is optimal when it's equal to
		// the streaming multiprocessor count of the device
		int blocks = Device.createDevice().getMultiprocessorCount();

		// Create Arguments
		HalfArg aMatrixArg = HalfArg.create(aMatrix);
		// Kernel expects a transposed B matrix so this has to be done here
		HalfArg bMatrixArg = HalfArg.createTransposed(y, z, bMatrix);
		FloatArg cMatrixArg = FloatArg.create(cMatrix);
		FloatArg dMatrixArg = FloatArg.createOutput(x * z);
		KernelArg mArg = IntArg.createValue(m);
		KernelArg nArg = IntArg.createValue(n);
		KernelArg kArg = IntArg.createValue(k);
		KernelArg alphaArg = FloatArg.createValue(alpha);
		KernelArg betaArg = FloatArg.createValue(beta);

		// Do the padding for each input matrix
		PaddingArg aMatrixArgPadding = PaddingArg.createMatrixPadding(aMatrixArg, x, y, m, k, 0);
		PaddingArg bMatrixArgPadding = PaddingArg.createMatrixPadding(bMatrixArg, z, y, n, k, 0);
		PaddingArg cMatrixArgPadding = PaddingArg.createMatrixPadding(cMatrixArg, x, z, m, n, 0);
		PaddingArg dMatrixArgPadding = PaddingArg.createMatrixPadding(dMatrixArg, x, z, m, n, 0);

		// Compiler options
		Options options = Options.createOptions("--gpu-architecture=compute_70");

		// Compile and launch Kernel
		KernelTime time = Executor.launch("fast_wmma_gemm", options, blocks, threads, aMatrixArgPadding,
				bMatrixArgPadding, cMatrixArgPadding, dMatrixArgPadding, mArg, nArg, kArg, alphaArg, betaArg);

		// Print Result
		System.out.println("Kernel fast_wmma_gemm launched " + time.toString());
		System.out.println();
		System.out.println("aMatrix:");
		MatrixUtils.printlnMatrix(aMatrixArg, y);
		System.out.println("bMatrix (transposed):");
		MatrixUtils.printlnMatrix(bMatrixArg, y);
		System.out.println("cMatrix:");
		MatrixUtils.printlnMatrix(cMatrixArg, z);
		System.out.println("resultmatrix:");
		MatrixUtils.printlnMatrix(dMatrixArg, z);
	}
}