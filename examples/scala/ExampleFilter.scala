object ExampleFilter {

    def main(args: Array[String]) : Unit = {
        Executor.loadLibary()

        var SIZE_INT = 4
        var numThreads = 16
        var numBlocks = 2

        //Testdata
        var length = 16;
        var src = new Array[Int](length);

        for (i <- 0 until length){
            src(i) = i;
        }

        println(src.mkString(", "))

        //Initialize Arguments
        val srcArg = ArrayArg.create(src, false)
        val outArg = ArrayArg.createOutput(length * SIZE_INT)
        val counterArg = ArrayArg.create(Array[Int](0), true);
        val nArg = ValueArg.create(length);

        //Create Program
        val kernelString = Utils.loadFile("filter_k.cu")
        val filter = Program.create(kernelString, "filter_k")

        //Create Kernel
        val filterKernel = filter.compile()

        //Launch Kernel
        filterKernel.launch(Array[KernelArg](outArg, counterArg, srcArg, nArg), numThreads, numBlocks)

        //Get Result
        var out = outArg.asIntArray()
        var counter = counterArg.asIntArray()(0)

        //Print Result
        println("Counter: " + counter);
        println("Result:  " + out.take(counter).mkString(", "))
    }
}