import java.io.File;
import java.io.IOException;
import java.nio.file.Files;

public class ExampleSaxpy {

    public static void main(String[] args) throws IOException{
        Executor.init();

        final int numThreads = 8;
        final int numBlocks = 8;

        //Testdata
        int n = numThreads*numBlocks;
        float a = 5.1f;
        float[] x = new float[n];
        float[] y = new float[n];
        for (int i = 0; i < n; ++i) {
            x[i] = i;
            y[i] = 2*i;
        }

        //Initialize Arguments
        KernelArg aArg, xArg, yArg, outArg, nArg;
        aArg = KernelArg.create(a);
        xArg = KernelArg.create(x, false);
        yArg = KernelArg.create(y, false);
        outArg = KernelArg.createOutput(n*4);
        nArg = KernelArg.create(n);

        //Create Program
        String kernelString = Utils.loadFile("../examples/kernels/saxpy.cu");
        Program saxpy = Program.create(kernelString, "saxpy");

        //Create Kernel
        Kernel saxpyKernel = saxpy.compile();

        //Compile and launch Kernel
        saxpyKernel.launch(new KernelArg[]{aArg, xArg, yArg, outArg, nArg}, numThreads, numBlocks);

        //Get Result
        float[] out = outArg.asFloatArray();

        boolean correct = true;
        for (int j = 0; j <  out.length; ++j) {
            float expected = x[j] * a + y[j];
            if ((expected - out[j]) > 10e-5) {
              correct = false;
              System.err.println("Exepected " + expected + " != Result " + out[j]);
             }
        }

         if (correct)
             System.out.println("Everything was calculated correctly!!!");
    }
}