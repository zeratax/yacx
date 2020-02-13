

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

public class ExampleSimpleGemm {
	public static void main(String[] args) throws IOException {
		//Load Libary
        Executor.loadLibary();
        
        //Testdata
        int n = 16;
        int m = 16;
        int k = 16;
        float alpha = 1f;
        float beta = 1f;
        float[] aMatrix = new float[n*m];
        float[] bMatrix = new float[m*k];
        float[] cMatrix = new float[n*k];
        for (int i = 0; i < n*m; i++) {
			aMatrix[i] = 1f;
		}
        for (int i = 0; i < m*k; i++) {
			bMatrix[i] = 1f;
		}
        for (int i = 0; i < n*k; i++) {
        	cMatrix[i] = 1f;
        }
        

        //Create Arguments
        HalfArg aMatrixArg = HalfArg.create(aMatrix);
        HalfArg bMatrixArg = HalfArg.create(bMatrix);
        FloatArg cMatrixArg = FloatArg.create(cMatrix);
        FloatArg dMatrixArg = FloatArg.createOutput(n*k);
        KernelArg mArg = IntArg.createValue(m);
        KernelArg nArg = IntArg.createValue(n);
        KernelArg kArg = IntArg.createValue(k);
        KernelArg alphaArg = FloatArg.createValue(alpha);
        KernelArg betaArg = FloatArg.createValue(beta);
        
        //Load Kernel as string
        String kernelString = Utils.loadFile("simple_wmma_gemm.cu");
        
        //Compiler option
        Options options = Options.createOptions("--gpu-architecture=compute_60");

        //Compile and launch Kernel
        KernelTime time = Executor.launch(kernelString, "simple_wmma_gemm", options, 1,1,1, 1,1,1,
        		aMatrixArg, bMatrixArg, cMatrixArg, dMatrixArg, mArg, nArg, kArg, alphaArg, betaArg);

        float[] dMatrix = dMatrixArg.asFloatArray();
        
        //Print Result
        System.out.println("Kernel simple_wmma_gemm launched" + time.toString());
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
