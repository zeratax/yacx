import scala.math.abs

object ExampleSaxpy {

    def main(args: Array[String]) : Unit = {
        //Load Libary
        Executor.loadLibary()

        //Testdata
        var numThreads = 4
        var numBlocks = 4

        var n = numThreads*numBlocks
        var a = 5.1f
        var x = new Array[Float](n)
        var y = new Array[Float](n)
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
        val kernelString = Utils.loadFile("saxpy.cu")
        val saxpy = Program.create(kernelString, "saxpy")

        //Create compiled Kernel
        val saxpyKernel = saxpy.compile()

        //Launch Kernel
        var executionTime = saxpyKernel.launch(numThreads, numBlocks, aArg, xArg, yArg, outArg, nArg)

        //Get Result
        var out = outArg.asFloatArray()

        //Print Result
        println("\nsaxpy-Kernel sucessfully launched:");
        println(executionTime);
        println("\nInput a: " + a);
        println("Input x: [" + x.mkString(", ") + "]");
        println("Input y: [" + y.mkString(", ") + "]");
        println("Result:  [" + out.mkString(", ") + "]");
    }
}