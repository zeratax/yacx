import java.io.IOException;
import java.util.Arrays;

import yacx.Executor;
import yacx.FloatArg;
import yacx.IntArg;
import yacx.Kernel;
import yacx.KernelArg;
import yacx.KernelTime;
import yacx.Program;
import yacx.Utils;

public class ExampleSaxpy {
    public static void main(String[] args) throws IOException {
        //Load Libary
        Executor.loadLibary();

        //Testdata
        final int numThreads = 4;
        final int numBlocks = 4;

        int n = numThreads*numBlocks;
        float a = 5.1f;
        float[] x = new float[n];
        float[] y = new float[n];
        for (int i = 0; i < n; i++) {
            x[i] = i;
            y[i] = 2*i;
        }

        //Initialize Arguments
        KernelArg aArg, nArg, xArg, yArg;
        FloatArg outArg;
        aArg = FloatArg.createValue(a);
        xArg = FloatArg.create(x);
        yArg = FloatArg.create(y);
        outArg = FloatArg.createOutput(n);
        nArg = IntArg.createValue(n);

        //Create Program
        String kernelString = Utils.loadFile("kernels/saxpy.cu");
        Program saxpy = Program.create(kernelString, "saxpy");

        //Create compiled Kernel
        Kernel saxpyKernel = saxpy.compile();

        //Launch Kernel
        KernelTime executionTime = saxpyKernel.launch(numThreads, numBlocks, aArg, xArg, yArg, outArg, nArg);

        //Get Result
        float[] out = outArg.asFloatArray();

        //Print Result
        System.out.println("\nsaxpy-Kernel sucessfully launched:");
        System.out.println(executionTime);
        System.out.println("\nInput a: " + a);
        System.out.println("Input x: " + Arrays.toString(x));
        System.out.println("Input y: " + Arrays.toString(y));
        System.out.println("Result:  " + Arrays.toString(out));
    }
}