import java.io.IOException;
import java.util.Arrays;

import yacx.Executor;
import yacx.IntArg;
import yacx.Kernel;
import yacx.KernelArg;
import yacx.LongArg;
import yacx.Program;
import yacx.Utils;

public class ExampleBlockReduce {
    public static void main(String[] args) throws IOException {
        //Load Libary
        Executor.loadLibary();

        //Testdata
        int arraySize = 32;

        final int numBlocks = 512;
        final int numThreads = Math.min((arraySize + numBlocks - 1) / numBlocks, 1024);

        long[] in = new long[arraySize];
        for (int i = 1; i <= in.length; i++){
            in[i-1] = i;
        }

        //Initialize Arguments
        LongArg inArg = LongArg.create(in, false);
        LongArg outArg = LongArg.createOutput(arraySize);
        KernelArg nArg = IntArg.createValue(arraySize);

        //Load kernelString
        String kernelString = Utils.loadFile("kernels/block_reduce.cu");
        //Create Program
        Program blockReduce = Program.create(kernelString, "deviceReduceKernel");

        //Create compiled Kernel
        Kernel blockReduceKernel = blockReduce.compile();

        //Launch Kernel
        blockReduceKernel.launch(numThreads, numBlocks, inArg, outArg, nArg);
        
        //New Input is Output from previous run
        inArg = LongArg.create(outArg.asLongArray());
        //Second launch
        blockReduceKernel.launch(numThreads, numBlocks, inArg, outArg, nArg);

        //Get Result
        long out = outArg.asLongArray()[0];

        //Print Result
        System.out.println("\nInput:");
        System.out.println(Arrays.toString(in));
        System.out.println("\nResult:   " + out);
        System.out.println("Expected: " + (arraySize * (arraySize+1))/2);
    }
}