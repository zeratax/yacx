import java.io.IOException
import yacx.Executor
import yacx.Executor.KernelArgCreator
import yacx.LongArg
import yacx.IntArg
import yacx.KernelArg
import yacx.Options

object ExampleReduceBenchmark {
  private val KB = 1024
  private val MB = 1024 * 1024

  def main(args: Array[Nothing]): Unit = {
    // Load library
    Executor.loadLibrary()

    val creator1 = new Executor.KernelArgCreator() {
      override def getDataLength(dataSizeBytes: Long): Int = (dataSizeBytes / LongArg.SIZE_BYTES).toInt

      override def getGrid0(dataLength: Int): Int = {
        val numThreads = getBlock0(0)
        Math.min((dataLength + numThreads - 1) / numThreads, 1024)
      }

      override def getBlock0(dataLength: Int) = 512

      override def createArgs(dataLength: Int): Array[KernelArg] = {
        val in = new Array[Long](dataLength)
        for (i <- 0 until in.length) {
          in(i) = i
        }
        val inArg = LongArg.create(in)
        val outArg = LongArg.createOutput(dataLength)
        val nArg = IntArg.createValue(dataLength)
        Array[KernelArg](inArg, outArg, nArg)
      }
    }
    val creator2 = new Executor.KernelArgCreator() {
      override def getDataLength(dataSizeBytes: Long): Int = (dataSizeBytes / LongArg.SIZE_BYTES).toInt

      override def getGrid0(dataLength: Int) = 1

      override def getBlock0(dataLength: Int) = 1024

      override def createArgs(dataLength: Int): Array[KernelArg] = {
        val blocks = Math.min((dataLength + 512 - 1) / 512, 1024)
        val in = new Array[Long](blocks)
        for (i <- 0 until in.length) {
          in(i) = i
        }
        val inArg = LongArg.create(in)
        val outArg = LongArg.createOutput(blocks)
        val nArg = IntArg.createValue(blocks)
        Array[KernelArg](inArg, outArg, nArg)
      }
    }
    val dataSizes = Array[Long](1 * KB, 4 * KB, 16 * KB, 64 * KB, 256 * KB, 1 * MB, 4 * MB, 16 * MB, 64 * MB, 256 * MB, 1024 * MB)

    // Options
    val options = Options.createOptions

    // Warm up
    Executor.benchmark("device_reduce", options, 30, creator1, 256 * MB)

    // Benchmark Reduce-Kernel
    val result = Executor.benchmark("device_reduce", options, 50, creator1, dataSizes)

    // Simulate second kernel call
    val result2 = Executor.benchmark("device_reduce", options, 50, creator2, dataSizes)

    // Add the average times of the second benchmark to the first
    val sum = result.addBenchmarkResult(result2)

    // Print out the final benchmark result
    println(sum)
  }
}