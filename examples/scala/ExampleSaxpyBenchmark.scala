object ExampleSaxpyBenchmark {
	private val KB = 1024
	
    def main(args: Array[String]) : Unit = {
    	//Load Libary
    	Executor.loadLibary()
    	
      val saxpy = Utils.loadFile("kernels/saxpy.cu")
        
      //Benchmark saxpy-Kernel
      println(Executor.benchmark(saxpy, "saxpy", Options.createOptions(), 10,
              new Executor.KernelArgCreator() {
                      val a = 5.1f
                			
          			      override def getDataLength(dataSizeBytes: Int) : Int = {
          						  return dataSizeBytes/FloatArg.SIZE_BYTES
          					  }
          
          					  override def getGrid0(dataLength: Int) : Int = {
          						  return dataLength
          					  }
          
            					override def getBlock0(dataLength: Int) : Int = {
            						return 1
            					}
            
            					override def createArgs(dataLength: Int) : Array[KernelArg] = {
            						val x = new Array[Float](dataLength)
            						val y = new Array[Float](dataLength)
            
            						for (i <- 0 until dataLength) {
            							x(i) = 1
            							y(i) = i
            						}
            
            						return Array[KernelArg] (FloatArg.createValue(a), FloatArg.create(x: _*), FloatArg.create(y: _*),
            								FloatArg.createOutput(dataLength), IntArg.createValue(dataLength))
            					}
				}, 1*KB, 4*KB, 8*KB, 1024*KB, 4096*KB, 16384*KB))
    }
}