import java.io.IOException;

import yacx.Executor;
import yacx.IntArg;
import yacx.KernelArg;
import yacx.Options;

public class ExampleFilterBenchmark {
	public static void main(String[] args) throws IOException {
		// Load Library
		Executor.loadLibrary();

		// Benchmark filter-Kernel
		System.out.println(Executor.benchmark("filter_k", Options.createOptions(), 10, new Executor.KernelArgCreator<Integer>() {

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
				int[] in = new int[dataLength];

				for (int i = 0; i < dataLength; i++) {
					in[i] = i;
				}

				return new KernelArg[] { IntArg.createOutput(dataLength), IntArg.create(new int[] { 0 }, true),
						IntArg.create(in), IntArg.create(dataLength) };
			}
		}, 1024, 2048, 4096, 131072));
	}
}