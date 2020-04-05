import java.io.IOException
import yacx.Executor
import yacx.FloatArg
import yacx.HalfArg
import yacx.IntArg
import yacx.KernelArg
import yacx.KernelTime
import yacx.Options
import yacx.PaddingArg
import yacx.Utils

object ExampleSimpleGEMM {
  // WMMA dimensions
  private val WMMA_M = 16
  private val WMMA_N = 16

  def main(args: Array[String]) : Unit = {
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

    // Get the next biggest multiple of 16 for each dimension
    val m = if (x % 16 == 0) x else (x / 16 + 1) * 16
    val k = if (y % 16 == 0) y else (y / 16 + 1) * 16
    val n = if (z % 16 == 0) z else (z / 16 + 1) * 16

    // Calculate block and grid dimensions
    // blockDim.x must be a multple of warpSize
    // 128x4 means we have 16 warps and a block zomputes a 64x64 output tile
    val blockDimX = 128
    val blockDimY = 4
    val gridDimX = (m + (WMMA_M * blockDimX / 32 - 1)) / (WMMA_M * blockDimX / 32)
    val gridDimY = (n + WMMA_N * blockDimY - 1) / (WMMA_N * blockDimY)

    // Create Arguments
    val aMatrixArg = HalfArg.create(aMatrix: _*)
    // Kernel expects a transposed B matrix so this has to be done here
    val bMatrixArg = HalfArg.createTransposed(bMatrix, y, z)
    val cMatrixArg = FloatArg.create(cMatrix: _*)
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
    val kernelString = Utils.loadFile("kernels/simple_wmma_gemm.cu")

    // Compiler options
    val options = Options.createOptions("--gpu-architecture=compute_70")

    // Compile and launch Kernel
    val time = Executor.launch(kernelString, "simple_wmma_gemm", options, gridDimX, gridDimY, 1, blockDimX, blockDimY, 1, aMatrixArgPadding, bMatrixArgPadding, cMatrixArgPadding, dMatrixArgPadding, mArg, nArg, kArg, alphaArg, betaArg)

    // Print Result
    println("Kernel simple_wmma_gemm launched " + time.toString)
    println
    println("aMatrix:")
    MatrixUtils.printlnMatrix(aMatrix, y)
    println("bMatrix:")
    MatrixUtils.printlnMatrix(bMatrix, z)
    println("cMatrix:")
    MatrixUtils.printlnMatrix(cMatrix, z)
    println("resultmatrix:")
    MatrixUtils.printlnMatrix(dMatrixArg.asFloatArray, z)
  }
}
