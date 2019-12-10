public class ExampleSimpleGEMM {
	
	// matrix dimensions
	private static final int M_GLOBAL = 672;
	private static final int N_GLOBAL = 672;
	private static final int K_GLOBAL = 672;
	
	// WMMA dimensions
	private static final int WMMA_M = 16;
	private static final int WMMA_N = 16;
	private static final int WMMA_K = 16;
	
	public static void main(String[] args) {
		//Load Libary
    Executor.loadLibary();
		
		// Initialize test matrices
		float[][] A = new float[M][K];
		float[][] B = new float[K][N];
		float[][] C = new float[M][N];
		
		// fill test matrices
		for (int i = 0; i < A.length; i++) {
			for (int j = 0; j < A[0].length; j++) {
				A[i][j] = i + j;
			}
		}
		for (int i = 0; i < B.length; i++) {
			for (int j = 0; j < B[0].length; j++) {
				B[i][j] = i + j;
			}
		}
		
		// initialize parameters alpha and beta
		final float alpha = 1.0;
		final float beta = 1.0;
		
		// blockDim.x must be a multple of warpSize
		// 128x4 means we have 16 warps and a block computes a 64x64 output tile
		int blockDimX = 128;
		int blockDimY = 4;

		int gridDimX = (M_GLOBAL + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
		int gridDimY = (N_GLOBAL + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);
	
		//Initialize Arguments
		FloatArg inArg_A = FloatArg.create(A, false);
		FloatArg inArg_B = FloatArg.create(B, false);
		FloatArg inArg_C = FloatArg.create(C, false);
		FloatArg outArg_D = FloatArg.createOutput(M_GLOBAL * N_GLOBAL);
		IntArg mArg = IntArg.create(M_GLOBAL);
		IntArg nArg = IntArg.create(N_GLOBAL);
		IntArg kArg = IntArg.create(K_GLOBAL);
		FloatArg alphaArg = FloatArg.create(alpha);
		FloatArg betaArg = FloatArg.create(beta);
		
		//Load kernelString
    String kernelString = Utils.loadFile("simple_wmma_gemm.cu");
    
    //Create Program
    Program simpleGEMM = Program.create(kernelString, "simple_wmma_gemm");
		
		//Create compiled Kernel
    Kernel simpleGEMMKernel = simpleGEMM.compile();

    //Compile and launch Kernel
    simpleGEMMKernel.launch(new KernelArg[]{inArg_A, inArg_B, inArg_C, outArg_D, mArg, nArg, kArg, alphaArg, betaArg},
								            gridDimX, gridDimY, blockDimX, blockDimY);
								
		//Get Result
		float[] out = outArg_D.asFloatArray();
		
		// Print Result
		System.out.println("\nResult:");
    System.out.println(Float.toString(out));
	}
}
