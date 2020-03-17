import yacx.Executor;
import ya


public class ExampleDotProduct {
    public static void main(String[] args ){
        //load library
        Executor.load();

        //testdata
        final int numThreads = 4;
        final int numBlocks = 1;

        int n = numThreads * numBlocks;
        float[] x = new float[n];
        float[] y = new float[n];

        for(int i = 0; i < n; i++){
            x[i] = i;
            y[i] = 2 * i;
        }

        //initalize arguments
        KernelArg nArg, xArg, yArg;
        FloatArg outArg;
        xArg = FloatArg.create(x);
        yArg = FloatArg.create(y);
        outArg = FloatArg.createOutput(n);
        nArg = IntArg.createValue(n);

        // Create Program
        String kernelString = Utils.loadFile("kernels/dotProduct.cu");
        Program dotProduct = Program.create(kernelString, "dotProduct");

        // Create compiled Kernel
        Kernel dotProductKernel = dotProduct.compile();

        // Launch Kernel
        KernelTime executionTime = dotProductKernel.launch(numThreads, numBlocks, xArg, yArg, outArg, nArg);

        // Get Result
        float[] out = outArg.asFloatArray();

        // Print Result
        System.out.println("\ndotProduct-Kernel sucessfully launched:");
        System.out.println(executionTime);
        System.out.println("\nInput x: " + Arrays.toString(x));
        System.out.println("Input y: " + Arrays.toString(y));
        System.out.println("Result:  " + Arrays.toString(out));


    }



}

