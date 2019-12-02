import java.io.IOException;
import java.util.Arrays;

public class ExampleBlockReduce {
    private static final int SIZE_LONG = 8;

    //TODO False Result
    public static void main(String[] args) throws IOException {
        //Load Libary
        Executor.loadLibary();

        //Testdata
        int arraySize = 16;

        final int numThreads = 512;
        final int numBlocks = Math.min((arraySize + numThreads - 1) / numThreads, 1024);

        long[] in = new long[arraySize];
        for (int i = 0; i < in.length; i++){
            in[i] = i;
        }

        //Initialize Arguments
        ArrayArg inArg = ArrayArg.create(in, false);
        ArrayArg outArg = ArrayArg.createOutput(arraySize * SIZE_LONG);
        KernelArg nArg = ValueArg.create(arraySize);

        //Load kernelString
        String kernelString = Utils.loadFile("block_reduce.cu");
        //Create Program
        Program blockReduce = Program.create(kernelString, "deviceReduceKernel");

        //Create compiled Kernel
        Kernel blockReduceKernel = blockReduce.compile();

        //Compile and launch Kernel
        blockReduceKernel.launch(new KernelArg[]{inArg, outArg, nArg}, numThreads, numBlocks);

        //Get Result
        long[] out = outArg.asLongArray();

        //Print Result
        System.out.println("\nInput:");
        System.out.println(Arrays.toString(in));
        System.out.println("\nResult:");
        System.out.println(Arrays.toString(out));
    }
}