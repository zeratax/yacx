import yacx.Executor;
import yacx.IntArg;
import yacx.Kernel;
import yacx.KernelArg;
import yacx.LongArg;
import yacx.Program;
import yacx.Utils;

object ExampleBlockReduce {
    def main(args: Array[String]) : Unit = {
        //Load Libary
        Executor.loadLibary();

        //Testdata
        val arraySize = 32;

        val numBlocks = 512;
        val numThreads = Math.min((arraySize + numBlocks - 1) / numBlocks, 1024);

        val in = new Array[Long](arraySize);
        for (i <- 1 until arraySize+1){
            in(i-1) = i;
        }

        //Initialize Arguments
        var inArg = LongArg.create(in, false);
        val outArg = LongArg.createOutput(arraySize);
        val nArg = IntArg.createValue(arraySize);

        //Load kernelString
        val kernelString = Utils.loadFile("kernels/block_reduce.cu");
        //Create Program
        val blockReduce = Program.create(kernelString, "deviceReduceKernel");

        //Create compiled Kernel
        val blockReduceKernel = blockReduce.compile();

        //Launch Kernel
        blockReduceKernel.launch(numThreads, numBlocks, inArg, outArg, nArg);
        
        //New Input is Output from previous run
        inArg = LongArg.create(outArg.asLongArray(): _*);
        //Second launch
        blockReduceKernel.launch(numThreads, numBlocks, inArg, outArg, nArg);

        //Get Result
        val out = outArg.asLongArray()(0);

        //Print Result
        println("\nInput:");
        println("[" + in.mkString(", ") + "]");
        println("\nResult:   " + out);
        println("Expected: " + (arraySize * (arraySize+1))/2);
    }
}