import java.io.IOException;
import java.util.Arrays;

public class ExampleReduce {
    //TODO False Result
    public static void main(String[] args) throws IOException {
        //Load Libary
        Executor.loadLibary();

        //Testdata
        int arraySize = 26368;

        final int numThreads = 512;
        final int numBlocks = Math.min((arraySize + numThreads - 1) / numThreads, 1024);

        long[] in = new long[arraySize];
        for (int i = 0; i < in.length; i++){
            in[i] = i;
        }

        //Initialize Arguments
        LongArg inArg = LongArg.create(in, false);
        LongArg outArg = LongArg.createOutput(arraySize);
        IntArg nArg = IntArg.create(arraySize);

        //Load kernelString
        String kernelString = Utils.loadFile("block_reduce.cu");
        //Create Program
        Program blockReduce = Program.create(kernelString, "deviceReduceKernel");

        //Create compiled Kernel
        Kernel blockReduceKernel = blockReduce.compile();

        //Compile and launch Kernel
        blockReduceKernel.launch(new KernelArg[]{inArg, outArg, nArg}, numBlocks, numThreads);

        //Next run
        inArg = LongArg.create(outArg.asLongArray());
        blockReduceKernel.launch(new KernelArg[]{inArg, outArg, nArg}, 1, 1024);

        //Get Result
        long out = outArg.asLongArray()[0];

        //Print Result
        System.out.println("\nInput:");
        System.out.println(Arrays.toString(in));
        System.out.println("\nResult:");
        System.out.println(Long.toString(out));
    }
}
