import java.io.IOException;
import java.util.Arrays;

public class ExampleBlockReduce {

    public static void main(String[] args) throws IOException{
        Executor.init();

        final int SIZE_LONG = 8;
        final int numThreads = 8;
        final int numBlocks = 8;

        //Testdata
        long val = 32;
        int n = 32;

        //Initialize Argument
        KernelArg valArg = ValueArg.create(val);
        ArrayArg outArg = ArrayArg.createOutput(n * SIZE_LONG);
        KernelArg nArg = ValueArg.create(n);

        //Load kernelString
        String kernelString = Utils.loadFile("block_reduce_Test.cu");
        //Headers included in kernelString
        Headers headers = Headers.createHeaders("/tmp/tmp.cTlciDtCIj/examples/kernels/block_reduce.h");
        //Create Program
        Program blockReduce = Program.create(kernelString, "blockReduceSumTest", headers);

        //Create Kernel
        Options options = Options.createOptions("-arch=compute_35");
        Kernel blockReduceKernel = blockReduce.compile(options);

        //Compile and launch Kernel
        blockReduceKernel.launch(new KernelArg[]{valArg, outArg, nArg}, numThreads, numBlocks);

        //Get Result
        long[] out = outArg.asLongArray();

        System.out.println(Arrays.toString(out));
    }
}