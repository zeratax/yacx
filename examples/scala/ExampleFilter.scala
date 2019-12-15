object ExampleFilter {
    def main(args: Array[String]) : Unit = {
        //Load Libary
        Executor.loadLibary()

        //Testdata
        val numThreads = 16
        val numBlocks = 1

        val n = 16
        val in = new Array[Int](n)

        for (i <- 0 until n){
            in(i) = i
        }

        //Initialize Arguments
        val srcArg = IntArg.create(in: _*)
        val outArg = IntArg.createOutput(n/2)
        val counterArg = IntArg.create(Array[Int](0), true)
        val nArg = IntArg.createValue(n)

        //Create Program
        val kernelString = Utils.loadFile("filter_k.cu")
        val filter = Program.create(kernelString, "filter_k")

        //Create compiled Kernel
        val filterKernel = filter.compile()

        //Launch Kernel
        val executionTime = filterKernel.launch(numThreads, numBlocks, outArg, counterArg, srcArg, nArg)

        //Get Result
        val out = outArg.asIntArray()
        val counter = counterArg.asIntArray()(0)

        //Print Result
        println("\nfilter-Kernel sucessfully launched:");
        println(executionTime);
        println("\nInput:          [" + in.mkString(", ") + "]");
        println("Result counter: " + counter);
        println("Result:         [" + out.mkString(", ") + "]");
    }
}