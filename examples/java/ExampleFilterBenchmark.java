import java.io.IOException;

import yacx.Executor;
import yacx.IntArg;
import yacx.KernelArg;
import yacx.Options;

public class ExampleFilterBenchmark {
	private final static int KB = 1024;
	
    public static void main(String[] args) throws IOException {
    	//Load Libary
    	Executor.loadLibary();
    	
        //Benchmark filter-Kernel
        System.out.println(Executor.benchmark("filter_k", Options.createOptions(), 10,
        		new Executor.KernelArgCreator() {
        	
		        	@Override
					public int getDataLength(int dataSizeBytes) {
						return (int) (dataSizeBytes/IntArg.SIZE_BYTES);
					}

					@Override
					public int getGrid0(int dataLength) {
						return dataLength;
					}
					
					@Override
					public int getBlock0(int dataLength) {
						return 1;
					}
					
					@Override
					public KernelArg[] createArgs(int dataLength) {
						int[] in = new int[dataLength];
						
						for (int i = 0; i < dataLength; i++) {
							in[i] = i;
						}
						
						return new KernelArg[] {IntArg.createOutput(dataLength), IntArg.create(new int[] {0}, true),
													IntArg.create(in), IntArg.create(dataLength)};
					}
				}, 1*KB, 4*KB, 8*KB, 1024*KB, 4096*KB, 131072*KB));
    }
}