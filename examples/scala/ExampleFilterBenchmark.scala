import yacx.Executor;
import yacx.IntArg;
import yacx.KernelArg;
import yacx.Options;
import yacx.Utils;

object ExampleFilterBenchmark {
	private val KB = 1024
	
    def main(args: Array[String]) : Unit = {
    	//Load Libary
    	Executor.loadLibary()
    	
      val saxpy = Utils.loadFile("filter_k.cu")
        
      //Benchmark filter-Kernel
      println(Executor.benchmark(saxpy, "filter_k", Options.createOptions(), 10,
              new Executor.KernelArgCreator() {
        
  					          override def getDataLength(dataSizeBytes: Int) : Int = {
            						return dataSizeBytes/IntArg.SIZE_BYTES
            					}
            
            					override def getGrid0(dataLength: Int) : Int = {
            						return dataLength
            					}
            					
            					override def getBlock0(dataLength: Int) : Int = {
            						return 1
            					}
            					
            					override def createArgs(dataLength: Int) : Array[KernelArg] = {
            						val in = new Array[Int](dataLength)
            						
            						for (i <- 0 until dataLength) {
            							in(i) = i
            						}
            						
            						return Array[KernelArg] (IntArg.createOutput(dataLength/2), IntArg.create(Array[Int](0), true),
            													IntArg.create(in: _*), IntArg.create(dataLength))
            					}
			}, 1*KB, 4*KB, 8*KB, 1024*KB, 4096*KB, 131072*KB))
    }
}