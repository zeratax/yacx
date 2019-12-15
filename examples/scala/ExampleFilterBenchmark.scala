
object ExampleFilterBenchmark {
	private final var KB = 1024
	
    def main(args: Array[String]) : Unit = {
    	//Load Libary
    	Executor.loadLibary()
    	
        var saxpy = Utils.loadFile("filter_k.cu")
        
        //Benchmark filter-Kernel
        println(Executor.benchmark(saxpy, "filter_k", Options.createOptions(), 10,
        		new Executor.KernelArgCreator() {
        	
					def getDataLength(dataSizeBytes: Int) : Int = {
						return dataSizeBytes/IntArg.SIZE_BYTES
					}

					def getGrid0(dataLength: Int) : Int = {
						return dataLength
					}
					
					def getBlock0(dataLength: Int) : Int = {
						return 1
					}
					
					def createArgs(dataLength: Int) : Array[KernelArg] = {
						var in = new Array[Int](dataLength)
						
						for (i <- 0 until dataLength) {
							in(i) = i
						}
						
						return Array[KernelArg] (IntArg.createOutput(dataLength/2), IntArg.create(Array[Int](0), true),
													IntArg.create(in: _*), IntArg.create(dataLength))
					}
				}, 1*KB, 4*KB, 8*KB, 1024*KB, 4096*KB, 131072*KB))
    }
}