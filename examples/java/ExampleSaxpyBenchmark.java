import java.io.IOException;

public class ExampleSaxpyBenchmark {
	private final static int KB = 1024;
	
    public static void main(String[] args) throws IOException {
    	//Load Libary
    	Executor.loadLibary();
    	
        String saxpy = Utils.loadFile("kernels/saxpy.cu");
        
        //Benchmark saxpy-Kernel
        System.out.println(Executor.benchmark(saxpy, "saxpy", Options.createOptions(), 10,
	            new Executor.KernelArgCreator() {
	                    final float a = 5.1f;
	        			
	        			@Override
						public int getDataLength(int dataSizeBytes) {
							return dataSizeBytes/FloatArg.SIZE_BYTES;
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
							float[] x = new float[dataLength];
							float[] y = new float[dataLength];
	
							for (int i = 0; i < dataLength; i++) {
								x[i] = 1;
								y[i] = i;
							}
	
							return new KernelArg[] {FloatArg.createValue(a), FloatArg.create(x), FloatArg.create(y),
									FloatArg.createOutput(dataLength), IntArg.createValue(dataLength)};
						}
				}, 1*KB, 4*KB, 8*KB, 1024*KB, 4096*KB, 16384*KB));
    }
}