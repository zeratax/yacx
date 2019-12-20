import yacx.Executor;
import yacx.FloatArg;
import yacx.IntArg;

object ExampleSaxpyExecutor {
    def main(args: Array[String]) : Unit = {
        //Load Libary
        Executor.loadLibary()

        //Create OutputArgument
        val n = 4
        val out = FloatArg.createOutput(n)

        //Compile and launch Kernel
        println("\n" + Executor.launch("saxpy", n, 1, FloatArg.createValue(5.1f), FloatArg.create(0,1,2,3),
                                FloatArg.create(2,2,4,4), out, IntArg.createValue(n)))

        //Print Result
        println("Result: [" + out.asFloatArray().mkString(", ") + "]")
    }
}