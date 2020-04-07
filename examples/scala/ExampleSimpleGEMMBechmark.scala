import java.io.IOException
import yacx.ArrayArg
import yacx.Executor
import yacx.KernelArg
import yacx.PaddingArg

object ExampleSimpleGEMMBenchmark {
  def main(args: Array[String]) : Unit = {

    val WMMA_M = 16
    val WMMA_N = 16

    // Load library
    Executor.loadLibrary()

    // Benchmark Simple-GEMM-Kernel
    new MatrixUtils.BenchmarkGEMM() {
      override def getGrid0(dim: Int): Int = {
        val m = if (dim % 16 == 0) dim
        else (dim / 16 + 1) * 16
        val blockDimX = getBlock0(0)
        (m + (WMMA_M * blockDimX / 32 - 1)) / (WMMA_M * blockDimX / 32)
      }

      override def getGrid1(dim: Int): Int = {
        val blockDimY = getBlock1(0)
        (dim + WMMA_N * blockDimY - 1) / (WMMA_N * blockDimY)
      }

      override def getBlock0(dim: Int) = 128

      override def getBlock1(dim: Int) = 4

      override def getPaddingDim(dim: Int): Int = {
        if (dim % 16 == 0)
          dim
        else
          (dim / 16 + 1) * 16
      }
    }.benchmark("simple_wmma_gemm")
  }
}