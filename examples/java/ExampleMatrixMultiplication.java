import java.io.IOException;
import java.util.Arrays;

import yacx.Executor;
import yacx.FloatArg;
import yacx.IntArg;
import yacx.Options;
import yacx.Utils;

public class ExampleMatrixMultiplication {
	public static void main(String[] args) throws IOException {
		//Load Libary
        Executor.loadLibary();

        //Create OutputArgument
        int n = 4;
        int m = 4;
        int k = 4;
        FloatArg aMatrix = FloatArg.create(1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f);
        FloatArg bMatrix = FloatArg.create(1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f);
        FloatArg cMatrix = FloatArg.createOutput(16);
        
        Options options = Options.createOptions("--gpu-architecture=compute_60");

        //Compile and launch Kernel
        String srcKernel = Utils.loadFile("simpleMatMul4.cu");
        System.out.println("\n" + Executor.launch(srcKernel, "matmul", options, 1,1,1, 1,1,1, aMatrix, bMatrix,
                                                 cMatrix, IntArg.createValue(n), IntArg.createValue(m),
                                                 IntArg.createValue(k)));

        //Print Result
        System.out.println("Result:         " + Arrays.toString(cMatrix.asFloatArray()));
	}
}
