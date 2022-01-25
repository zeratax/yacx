import java.io.IOException;

import yacx.Executor;
import yacx.FloatArg;
import yacx.IntArg;
import yacx.KernelArg;
import yacx.Options;

public class ExampleSaxpyBenchmark {

	public static void main(String[] args) throws IOException {
		// Load library
		Executor.loadLibrary();

		// Benchmark saxpy-Kernel
		System.out.println(Executor.benchmark("saxpy", Options.createOptions(), 10, new Executor.KernelArgCreator<Integer>() {
			final float a = 5.1f;

			@Override
			public int getGrid0(Integer dataLength) {
				return dataLength;
			}

			@Override
			public int getBlock0(Integer dataLength) {
				return 1;
			}

			@Override
			public KernelArg[] createArgs(Integer dataLength) {
				float[] x = new float[dataLength];
				float[] y = new float[dataLength];

				for (int i = 0; i < dataLength; i++) {
					x[i] = 1;
					y[i] = i;
				}

				return new KernelArg[] { FloatArg.createValue(a), FloatArg.create(x), FloatArg.create(y),
						FloatArg.createOutput(dataLength), IntArg.createValue(dataLength) };
			}
		}, 1024, 4096, 8192, 1024 * 1024, 4096 * 1024, 16384 * 1024));
	}
}