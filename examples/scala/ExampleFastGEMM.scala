import java.io.IOException
import yacx.Device
import yacx.Devices
import yacx.Executor
import yacx.FloatArg
import yacx.HalfArg
import yacx.IntArg
import yacx.KernelArg
import yacx.KernelTime
import yacx.Options
import yacx.PaddingArg
import yacx.Utils

object ExampleFastGEMM {
  def main(args: Array[Nothing]): Unit = {
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

    // Testdata
    val x = 4
    val y = 3
    val z = 2
    val alpha = 1f
    val beta = 1f
    val aMatrix = new Array[Float](x * y)
    val bMatrix = new Array[Float](y * z)
    val cMatrix = new Array[Float](x * z)

    for (i <- 0 until aMatrix.length) {
      aMatrix(i) = i + 1
    }
    for (i <- 0 until bMatrix.length) {
      bMatrix(i) = x * y + i + 1
    }
    for (i <- 0 until cMatrix.length) {
      cMatrix(i) = 2 * (i + 1)
    }

    // Get the next biggest multiple of 128 for each dimension
    val m = if (x % 128 == 0) x else (x / 128 + 1) * 128
    val k = if (y % 128 == 0) y else (y / 128 + 1) * 128
    val n = if (z % 128 == 0) z else (z / 128 + 1) * 128
    // Get Device
    val device = Devices.findDevice

    // 8 Warps = 256 Threads per Block are required for the kernel to work
    val threads = 32 * 8

    // The amount of blocks can be freely chosen but is optimal when it's equal to
    // the streaming multiprocessor count of the device
    val blocks = device.getMultiprocessorCount

    // Compute the right amount of shared memory to request.
    // We need shared memory to hold per-CTA C and D matrix tiles, and to cache
    // per-CTA chunks of the A and B matrices.
    // Therefore, the right amount to request is the maximum of those two numbers.
    val SHMEM_SZ = Math.max(HalfArg.SIZE_BYTES * (BLOCK_COL_TILES * M) * (CHUNK_K * K + SKEW_HALF) * 2,
      M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * N * (BLOCK_COL_WARPS * WARP_COL_TILES) * FloatArg.SIZE_BYTES)

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

    // Create Arguments
    val aMatrixArg = HalfArg.create(aMatrix)

    // Kernel expects a transposed B matrix so this has to be done here
    val bMatrixArg = HalfArg.createTransposed(bMatrix, y, z)
    val cMatrixArg = FloatArg.create(cMatrix)
    val dMatrixArg = FloatArg.createOutput(x * z)
    val mArg = IntArg.createValue(m)
    val nArg = IntArg.createValue(n)
    val kArg = IntArg.createValue(k)
    val alphaArg = FloatArg.createValue(alpha)
    val betaArg = FloatArg.createValue(beta)

    // Do the padding for each input matrix if necessary
    val aMatrixArgPadding = if (x == m && y == k) aMatrixArg
    else PaddingArg.createMatrixPadding(aMatrixArg, x, y, m, k, 0)

    val bMatrixArgPadding = if (z == n && y == k) bMatrixArg
    else PaddingArg.createMatrixPadding(bMatrixArg, z, y, n, k, 0)

    val cMatrixArgPadding = if (x == m && z == n) cMatrixArg
    else PaddingArg.createMatrixPadding(cMatrixArg, x, z, m, n, 0)

    val dMatrixArgPadding = if (x == m && z == n) dMatrixArg
    else PaddingArg.createMatrixPadding(dMatrixArg, x, z, m, n, 0)

    // Load Kernel as string
    val kernelString = Utils.loadFile("kernels/fast_wmma_gemm.cu")

    // Compiler options
    val options = Options.createOptions("--gpu-architecture=compute_70")

    // Compile and launch Kernel
    val time = Executor.launch(kernelString, "fast_wmma_gemm", options, blocks, 1, 1, threads, 1, 1, SHMEM_SZ, aMatrixArgPadding, bMatrixArgPadding, cMatrixArgPadding, dMatrixArgPadding, mArg, nArg, kArg, alphaArg, betaArg)

    // Print Result
    System.out.println("Kernel fast_wmma_gemm launched " + time.toString)
    System.out.println
    System.out.println("aMatrix:")
    MatrixUtils.printlnMatrix(aMatrix, y)
    System.out.println("bMatrix:")
    MatrixUtils.printlnMatrix(bMatrix, z)
    System.out.println("cMatrix:")
    MatrixUtils.printlnMatrix(cMatrix, z)
    System.out.println("resultmatrix:")
    MatrixUtils.printlnMatrix(dMatrixArg.asFloatArray, z)
  }
}