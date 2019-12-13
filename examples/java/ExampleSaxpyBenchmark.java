import java.io.IOException;

public class ExampleSaxpyBenchmark {
	private final static int KB = 1024;
	
    public static void main(String[] args) throws IOException {
    	//Load Libary
    	Executor.loadLibary();
    	
        String saxpy = Utils.loadFile("saxpy.cu");
        
        //Benchmark saxpy-Kernel
        System.out.println(Executor.benchmark(saxpy, "saxpy", Options.createOptions(), 10,
        		new Executor.KernelArgCreator() {
        			float a = 5.1f;

					@Override
					public int getGrid0(int dataSize) {
						return dataSize;
					}

					@Override
					public int getBlock0(int dataSize) {
						return 1;
					}

					@Override
					public KernelArg[] createArgs(int dataSize) {
						int n = dataSize / 4;
						float[] x = new float[n];
						float[] y = new float[n];

						for (int i = 0; i < n; i++) {
							x[i] = 1;
							y[i] = i;
						}

						return new KernelArg[] {FloatArg.create(a), FloatArg.create(x), FloatArg.create(y),
								FloatArg.createOutput(n), IntArg.create(n)};
					}
				}, 1*KB, 4*KB, 8*KB, 1024*KB, 4096*KB, 16384*KB));
    }
}