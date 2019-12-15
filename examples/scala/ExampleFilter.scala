object ExampleFilter {
    def main(args: Array[String]) : Unit = {
        //Load Libary
        Executor.loadLibary()

        //Testdata
        var numThreads = 16
        var numBlocks = 1

        var n = 16
        var in = new Array[Int](n)

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
        var executionTime = filterKernel.launch(numThreads, numBlocks, outArg, counterArg, srcArg, nArg)

        //Get Result
        var out = outArg.asIntArray()
        var counter = counterArg.asIntArray()(0)

        //Print Result
        println("\nfilter-Kernel sucessfully launched:");
        println(executionTime);
        println("\nInput:          [" + in.mkString(", ") + "]");
        println("Result counter: " + counter);
        println("Result:         [" + out.mkString(", ") + "]");
    }
}