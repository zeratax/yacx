import java.io.IOException
import yacx.ArrayArg
import yacx.Device
import yacx.Devices
import yacx.Executor
import yacx.FloatArg
import yacx.HalfArg
import yacx.KernelArg
import yacx.PaddingArg

object ExampleFastGEMMBenchmark {
  def main(args: Array[String]) : Unit = {
    // Constants for shared memory calculation
    val M = 16
    val N = 16
    val K = 16

    // If you change this, don't forget to adjust the SHARED_MEMORY_LIMIT_64K in the
    // kernel, too.
    val SHARED_MEMORY_LIMIT_64K = false
    val BLOCK_ROW_WARPS = 2
    val BLOCK_COL_WARPS = 4
    val WARP_ROW_TILES = 4
    val WARP_COL_TILES = 2
    val BLOCK_COL_TILES = WARP_COL_TILES * BLOCK_COL_WARPS
    val CHUNK_K = if (SHARED_MEMORY_LIMIT_64K) 4 else 8
    val SKEW_HALF = 8

    // Load library
    Executor.loadLibrary()

    // Get Device
    val device = Devices.findDevice

    // Compute required shared memory size
    val SHMEM_SZ = Math.max(HalfArg.SIZE_BYTES * (BLOCK_COL_TILES * M) * (CHUNK_K * K + SKEW_HALF) * 2, M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * N * (BLOCK_COL_WARPS * WARP_COL_TILES) * FloatArg.SIZE_BYTES)

    // Calculate and print out the required and available shared memory size
    val required = SHMEM_SZ / 1024
    val available = device.getSharedMemPerMultiprocessor / 1024
    System.out.println("Required shared memory size per multiprocessor: " + required + " KB")
    System.out.println("Available shared memory size per multiprocessor: " + available + " KB")

    // Check if there's enough shared memory per block available on the device for
    // this kernel
    if (required > available) {
      System.out.println("Not enough shared memory per block available on the device for this kernel! Abort!")
      System.out.println("Please use the simple GEMM kernel instead or increase the amount of shared memory per block if possible!")
      System.exit(1)
    }

    // Benchmark Simple-GEMM-Kernel
    new MatrixUtils.BenchmarkGEMM() {
      override def getGrid0(dim: Int): Int = device.getMultiprocessorCount

      override def getBlock0(dim: Int): Int = 32 * 8

      override def getSharedMemory(dataSizeBytes: Long): Long = SHMEM_SZ

      override def getPaddingDim(dim: Int): Int = {
        if (dim % 128 == 0)
          dim
        else
          (dim / 128 + 1) * 128
      }
    }.benchmark("fast_wmma_gemm")
  }
}
