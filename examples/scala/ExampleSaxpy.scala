object ExampleSaxpy {

    def main(args: Array[String]) : Unit = {
        Executor.init()

        var numThreads = 8
        var numBlocks = 8

        //Testdata
        var n = numThreads*numBlocks
        var a = 5.1f
        var x = new Array[Float](n)
        var y = new Array[Float](n)
        for (i <- 0 until n) {
            x(i) = i
            y(i) = 2*i
        }

        //Initialize Arguments
        val aArg = ValueArg.create(a)
        val xArg = ArrayArg.create(x, false)
        val yArg = ArrayArg.create(y, false)
        val outArg = ArrayArg.createOutput(n*4)
        val nArg = ValueArg.create(n)

        //Create Program
        val kernelString = Utils.loadFile("saxpy.cu")
        val saxpy = Program.create(kernelString, "saxpy")

        //Create Kernel
        val saxpyKernel = saxpy.compile()

        //Launch Kernel
        saxpyKernel.launch(Array[KernelArg](aArg, xArg, yArg, outArg, nArg), numThreads, numBlocks)

        //Get Result
        var out = outArg.asFloatArray()

        var correct = true
        for (j <- 0 until n) {
            var expected = x(j) * a + y(j)
            if ((expected - out(j)) > 10e-5) {
              correct = false
              println("Exepected " + expected + " != Result " + out(j))
             }
        }

         if (correct)
             println("Everything was calculated correctly!!!")
    }
}