import scala.math.abs

object ExampleSaxpy {

    def main(args: Array[String]) : Unit = {
        //Load Libary
        Executor.loadLibary();

        //Testdata
        var numThreads = 8
        var numBlocks = 8

        var n = numThreads*numBlocks
        var a = 5.1f
        var x = new Array[Float](n)
        var y = new Array[Float](n)
        for (i <- 0 until n) {
            x(i) = i
            y(i) = 2*i
        }

        //Initialize Arguments
        val aArg = FloatArg.create(a)
        val xArg = FloatArg.create(x, false)
        val yArg = FloatArg.create(y, false)
        val outArg = FloatArg.createOutput(n)
        val nArg = IntArg.create(n)

        //Create Program
        val kernelString = Utils.loadFile("saxpy.cu")
        val saxpy = Program.create(kernelString, "saxpy")

        //Create compiled Kernel
        val saxpyKernel = saxpy.compile()

        //Launch Kernel
        saxpyKernel.launch(numThreads, numBlocks, aArg, xArg, yArg, outArg, nArg)

        //Get Result
        var out = outArg.asFloatArray()

        //Check Result
        var correct = true
        for (j <- 0 until n) {
            var expected = x(j) * a + y(j)
            if (abs(expected - out(j)) > 10e-5) {
              correct = false
              println("Exepected " + expected + " != Result " + out(j))
             }
        }

         if (correct)
             println("\nEverything was calculated correctly!!!")
    }
}