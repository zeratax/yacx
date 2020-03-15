import java.io.IOException;
import java.util.Arrays;

import yacx.Executor;
import yacx.FloatArg;
import yacx.HalfArg;
import yacx.IntArg;
import yacx.KernelArg;
import yacx.KernelTime;
import yacx.Options;
import yacx.PaddingArg;
import yacx.Utils;

public class ExampleSimpleGEMM {
	// WMMA dimensions
	private final static int WMMA_M = 16;
	private final static int WMMA_N = 16;

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
			cMatrix[i] = 1f;
		}

		// Get the next biggest multiple of 16 for each dimension
		int m = (x % 16 == 0) ? x : (x / 16 + 1) * 16;
		int k = (y % 16 == 0) ? y : (y / 16 + 1) * 16;
		int n = (z % 16 == 0) ? z : (z / 16 + 1) * 16;

		// Calculate block and grid dimensions
		// blockDim.x must be a multple of warpSize
		// 128x4 means we have 16 warps and a block zomputes a 64x64 output tile
		int blockDimX = 128;
		int blockDimY = 4;

		int gridDimX = (m + (WMMA_M * blockDimX / 32 - 1)) / (WMMA_M * blockDimX / 32);
		int gridDimY = (n + WMMA_N * blockDimY - 1) / (WMMA_N * blockDimY);

		// Create Arguments
		HalfArg aMatrixArg = HalfArg.create(aMatrix);
		// Kernel expects a transposed B matrix so this has to be done here
		//HalfArg bMatrixArg = HalfArg.createTransposed(z, bMatrix);
		HalfArg bMatrixArg = HalfArg.create(bMatrix);
		FloatArg cMatrixArg = FloatArg.create(cMatrix);
		FloatArg dMatrixArg = FloatArg.createOutput(x * z);
		KernelArg mArg = IntArg.createValue(m);
		KernelArg nArg = IntArg.createValue(n);
		KernelArg kArg = IntArg.createValue(k);
		KernelArg alphaArg = FloatArg.createValue(alpha);
		KernelArg betaArg = FloatArg.createValue(beta);

		// Do the padding for each input matrix
		//TODO Padding falls (x,y,z) = (m,n,k)
		PaddingArg aMatrixArgPadding = PaddingArg.createMatrixPadding(aMatrixArg, x, y, m, k, 0);
		PaddingArg bMatrixArgPadding = PaddingArg.createMatrixPadding(bMatrixArg, y, z, n, k, 0);
		PaddingArg cMatrixArgPadding = PaddingArg.createMatrixPadding(cMatrixArg, x, z, m, n, 0);
		PaddingArg dMatrixArgPadding = PaddingArg.createMatrixPadding(dMatrixArg, x, z, m, n, 0);

		// Load Kernel as string
		String kernelString = Utils.loadFile("kernels/simple_wmma_gemm.cu");

		// Compiler options
		Options options = Options.createOptions("--gpu-architecture=compute_70");

		// Compile and launch Kernel
		KernelTime time = Executor.launch(kernelString, "simple_wmma_gemm", options, gridDimX, gridDimY, 1, blockDimX,
				blockDimY, 1, aMatrixArgPadding, bMatrixArgPadding, cMatrixArgPadding, dMatrixArgPadding, mArg, nArg,
				kArg, alphaArg, betaArg);

		// Print Result
		System.out.println("Kernel simple_wmma_gemm launched " + time.toString());
		System.out.println();
		System.out.println("aMatrix:");
		MatrixUtils.printlnMatrix(aMatrixArg, y);
		System.out.println("bMatrix:");
		MatrixUtils.printlnMatrix(bMatrixArg, z);
		System.out.println("cMatrix:");
		MatrixUtils.printlnMatrix(cMatrixArg, z);
		System.out.println("resultmatrix:");
		MatrixUtils.printlnMatrix(dMatrixArg, z);
	}
}
