object ExampleSaxpy {
    def main(args: Array[String]) : Unit = {
        //Load Libary
        Executor.loadLibary()

        //Testdata
        val numThreads = 4
        val numBlocks = 4

        val n = numThreads*numBlocks
        val a = 5.1f
        val x = new Array[Float](n)
        val y = new Array[Float](n)
        for (i <- 0 until n) {
            x(i) = i
            y(i) = 2*i
        }

        //Initialize Arguments
        val aArg = FloatArg.createValue(a)
        val xArg = FloatArg.create(x: _*)
        val yArg = FloatArg.create(y: _*)
        val outArg = FloatArg.createOutput(n)
        val nArg = IntArg.createValue(n)

        //Create Program
        val kernelString = Utils.loadFile("kernels/saxpy.cu")
        val saxpy = Program.create(kernelString, "saxpy")

        //Create compiled Kernel
        val saxpyKernel = saxpy.compile()

        //Launch Kernel
        val executionTime = saxpyKernel.launch(numThreads, numBlocks, aArg, xArg, yArg, outArg, nArg)

        //Get Result
        val out = outArg.asFloatArray()

        //Print Result
        println("\nsaxpy-Kernel sucessfully launched:");
        println(executionTime);
        println("\nInput a: " + a);
        println("Input x: [" + x.mkString(", ") + "]");
        println("Input y: [" + y.mkString(", ") + "]");
        println("Result:  [" + out.mkString(", ") + "]");
    }
}