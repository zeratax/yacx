import java.io.IOException
import yacx.Executor
import yacx.IntArg
import yacx.KernelArg
import yacx.KernelTime
import yacx.LongArg

object ExampleReduce {
  def main(args: Array[String]) : Unit = {
    // Load library
    Executor.loadLibrary()

    // Testdata
    val arraySize = 49631
    val numThreads = 512
    val numBlocks = Math.min((arraySize + numThreads - 1) / numThreads, 1024)
    val in = new Array[Long](arraySize)
    for (i <- 0 until in.length) {
      in(i) = i + 1
    }

    // Initialize Arguments
    val inArg = LongArg.create(in: _*)
    val outArg = LongArg.createOutput(arraySize)
    val nArg = IntArg.createValue(arraySize)

    // Launch Kernel
    val time = Executor.launch("device_reduce", numBlocks, numThreads, inArg, outArg, nArg)

    // New Input is Output from previous run
    outArg.setUpload(true)
    outArg.setDownload(false)
    // Use Input from previous run as Output
    inArg.setUpload(false)
    inArg.setDownload(true)

    // Second launch
    time.addKernelTime(Executor.launch("device_reduce", 1, 1024, outArg, inArg, nArg))

    // Get Result
    val out = inArg.asLongArray()(0)
    val expected = (arraySize.toLong * (arraySize.toLong + 1)) / 2

    // Print Result
    println("Kernel deviceReduce launched " + time.toString)
    println
    println("\nInput: 1..." + arraySize)
    println("\nResult:   " + out)
    println("Expected: " + expected)
  }
}
