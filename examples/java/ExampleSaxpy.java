import java.io.IOException;

public class ExampleSaxpy {
    private static final double DELTA = 10e-5;

    public static void main(String[] args) throws IOException {
        //Load Libary
        Executor.loadLibary();

        //Testdata
        final int numThreads = 8;
        final int numBlocks = 8;

        int n = numThreads*numBlocks;
        float a = 5.1f;
        float[] x = new float[n];
        float[] y = new float[n];
        for (int i = 0; i < n; ++i) {
            x[i] = i;
            y[i] = 2*i;
        }

        //Initialize Arguments
        ValueArg aArg, nArg;
        ArrayArg xArg, yArg, outArg;
        aArg = ValueArg.create(a);
        xArg = ArrayArg.create(x, false);
        yArg = ArrayArg.create(y, false);
        outArg = ArrayArg.createOutput(n*4);
        nArg = ValueArg.create(n);

        //Create Program
        String kernelString = Utils.loadFile("saxpy.cu");
        Program saxpy = Program.create(kernelString, "saxpy");

        //Create compiled Kernel
        Kernel saxpyKernel = saxpy.compile();

        //Compile and launch Kernel
        saxpyKernel.launch(new KernelArg[]{aArg, xArg, yArg, outArg, nArg}, numThreads, numBlocks);

        //Get Result
        float[] out = outArg.asFloatArray();

        //Check Result
        boolean correct = true;
        for (int j = 0; j <  out.length; ++j) {
            float expected = x[j] * a + y[j];
            if ((expected - out[j]) > DELTA) {
              correct = false;
              System.err.println("Exepected " + expected + " != Result " + out[j]);
             }
        }

         if (correct)
             System.out.println("Everything was calculated correctly!!!");
    }
}