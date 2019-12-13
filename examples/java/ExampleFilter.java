import java.io.IOException;
import java.util.Arrays;

public class ExampleFilter {
    public static void main(String[] args) throws IOException {
        //Load Libary
        Executor.loadLibary();

        //Testdata
        final int numThreads = 16;
        final int numBlocks = 1;

        int n = numThreads*numBlocks;
        int[] in = new int[n];
        for (int i = 0; i < n; i++) {
            in[i] = i;
        }

        //Initialize Arguments
        IntArg outArg, counterArg, inArg, nArg;
        outArg = IntArg.createOutput(n/2);
        counterArg = IntArg.createOutput(1);
        inArg = IntArg.create(in);
        nArg = IntArg.create(n);

        //Create Program
        String kernelString = Utils.loadFile("filter_k.cu");
        Program filter = Program.create(kernelString, "filter_k");

        //Create compiled Kernel
        Kernel filterKernel = filter.compile();

        //Compile and launch Kernel
        KernelTime t = filterKernel.launch(numThreads, numBlocks, outArg, counterArg, inArg, nArg);

        //Get Result
        int[] out = outArg.asIntArray();
        int counter = counterArg.asIntArray()[0];

        //Print Result
        System.out.println("\nfilter-Kernel (result = {x | x % 2 == 1}) sucessfully launched:");
        System.out.println(t);
        System.out.println("Input:          " + Arrays.toString(in));
        System.out.println("Result counter: " + counter);
        System.out.println("Result:         " + Arrays.toString(out));
    }
}