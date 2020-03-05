import java.io.IOException;
import java.util.Arrays;

import yacx.Executor;
import yacx.FloatArg;
import yacx.HalfArg;
import yacx.IntArg;
import yacx.KernelArg;
import yacx.KernelTime;
import yacx.Options;
import yacx.Utils;

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
		float[] aMatrix = new float[x * z];
		float[] bMatrix = new float[z * y];
		float[] cMatrix = new float[x * y];
		for (int i = 0; i < x * z; i++) {
			aMatrix[i] = 1f;
		}
		for (int i = 0; i < z * y; i++) {
			bMatrix[i] = 1f;
		}
		for (int i = 0; i < x * y; i++) {
			cMatrix[i] = 1f;
		}

		// Get the next biggest multiple of 128 for each dimension
		int m = (x % 128 == 0) ? x : (x / 128 + 1) * 128;
		int n = (y % 128 == 0) ? y : (y / 128 + 1) * 128;
		int k = (z % 128 == 0) ? z : (z / 128 + 1) * 128;

        // 8 Warps = 256 Threads per Block are required for the kernel to work
		int threads = 32 * 8;
        // The amount of blocks can be freely chosen but is optimal when it's equal to the streaming multiprocessor count of the device
        // (not yet implemented, 80 is the amount of SMs on a Tesla V100 like on the PALMA)
        int blocks = 80;

		// Create Arguments
		HalfArg aMatrixArg = HalfArg.create(aMatrix);
		// Kernel expects a transposed B matrix so this has to be done here
		HalfArg bMatrixArg = HalfArg.createTransposed(bMatrix, y);
		FloatArg cMatrixArg = FloatArg.create(cMatrix);
		FloatArg dMatrixArg = FloatArg.createOutput(m * n);
		KernelArg mArg = IntArg.createValue(m);
		KernelArg nArg = IntArg.createValue(n);
		KernelArg kArg = IntArg.createValue(k);
		KernelArg alphaArg = FloatArg.createValue(alpha);
		KernelArg betaArg = FloatArg.createValue(beta);

		// Do the padding for each input matrix
		PaddingArg aMatrixArgPadding = PaddingArg.createMatrixPadding(aMatrixArg, x, z, m, k, 0);
		PaddingArg bMatrixArgPadding = PaddingArg.createMatrixPadding(bMatrixArg, y, z, n, k, 0);
		PaddingArg cMatrixArgPadding = PaddingArg.createMatrixPadding(cMatrixArg, x, y, m, n, 0);

		// Load Kernel as string
		String kernelString = Utils.loadFile("kernels/fast_wmma_gemm.cu");

		// Compiler options
		Options options = Options.createOptions("--gpu-architecture=compute_70");

		// Compile and launch Kernel
		KernelTime time = Executor.launch(kernelString, "fast_wmma_gemm", options, blocks, threads,
				blockDimY, 1, aMatrixArgPadding, bMatrixArgPadding, cMatrixArgPadding, dMatrixArg, mArg, nArg, kArg, alphaArg, betaArg);

		float[] dMatrix = dMatrixArg.asFloatArray();

		// Print Result
		System.out.println("Kernel fast_wmma_gemm launched " + time.toString());
		System.out.println();
		System.out.println("aMatrix:");
		System.out.println(Arrays.toString(aMatrix));
		System.out.println("bMatrix:");
		System.out.println(Arrays.toString(bMatrix));
		System.out.println("cMatrix:");
		System.out.println(Arrays.toString(cMatrix));
		System.out.println("resultmatrix:");
		System.out.println(Arrays.toString(dMatrix));
	}
}