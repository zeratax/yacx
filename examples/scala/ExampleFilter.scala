object ExampleFilter {
    def main(args: Array[String]) : Unit = {
        //Load Libary
        Executor.loadLibary()

        //Testdata
        var numThreads = 16
        var numBlocks = 1

        var length = 16;
        var src = new Array[Int](length);

        for (i <- 0 until length){
            src(i) = i;
        }

        //Initialize Arguments
        val srcArg = IntArg.create(src: _*)
        val outArg = IntArg.createOutput(length)
        val counterArg = IntArg.createOutput(1);
        val nArg = IntArg.create(length);

        //Create Program
        val kernelString = Utils.loadFile("filter_k.cu")
        val filter = Program.create(kernelString, "filter_k")

        //Create compiled Kernel
        val filterKernel = filter.compile()

        //Launch Kernel
        filterKernel.launch(numThreads, numBlocks, outArg, counterArg, srcArg, nArg)

        //Get Result
        var out = outArg.asIntArray()
        var counter = counterArg.asIntArray()(0)

        //Print Result
        println("\nInput:\n[" + src.mkString(", ") + "]")
        println("\nResult:")
        println("Counter: " + counter);
        println("Result:\n[" + out.take(counter).mkString(", ") + "]")
    }
}