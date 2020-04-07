import java.io.IOException
import java.util.Arrays
import yacx.Executor
import yacx.FloatArg
import yacx.KernelTime

object ExampleDotProduct {
  def main(args: Array[String]) : Unit = {
    // Load library
    Executor.loadLibrary()

    // Testdata
    val numberElements = 9
    val x = new Array[Float](numberElements)
    val y = new Array[Float](numberElements)
    for (i <- 0 until numberElements) {
      x(i) = i
      y(i) = 2 * i
    }

    // Initalize arguments
    val xArg = FloatArg.create(x: _*)
    val yArg = FloatArg.create(y: _*)
    val outArg = FloatArg.createOutput(1)

    // Compile and Launch
    val executionTime = Executor.launch("dotProduct", 1, numberElements, xArg, yArg, outArg)

    // Get Result
    val result = outArg.asFloatArray()(0)

    // Print Result
    println("\ndotProduct-Kernel sucessfully launched:")
    println(executionTime)
    println("\nInput x: " + Arrays.toString(x))
    println("Input y: " + Arrays.toString(y))
    println("Result:  " + result)
  }
}
