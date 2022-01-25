import java.io.IOException;

import yacx.Executor;
import yacx.Executor.KernelArgCreator;
import yacx.LongArg;
import yacx.IntArg;
import yacx.KernelArg;
import yacx.Options;

public class ExampleReduceBenchmark {

	public static void main(String[] args) throws IOException {
		// Load library
		Executor.loadLibrary();

		KernelArgCreator<Integer> creator1 = new Executor.KernelArgCreator<Integer>() {

			@Override
			public int getGrid0(Integer dataLength) {
				int numThreads = getBlock0(0);
				return Math.min((dataLength + numThreads - 1) / numThreads, 1024);
			}

			@Override
			public int getBlock0(Integer dataLength) {
				return 512;
			}

			@Override
			public KernelArg[] createArgs(Integer dataLength) {
				long[] in = new long[dataLength];
				for (int i = 0; i < in.length; i++) {
					in[i] = i;
				}

				LongArg inArg = LongArg.create(in);
				LongArg outArg = LongArg.createOutput(dataLength);
				KernelArg nArg = IntArg.createValue(dataLength);

				return new KernelArg[] { inArg, outArg, nArg };
			}
		};

		KernelArgCreator<Integer> creator2 = new Executor.KernelArgCreator<Integer>() {

			@Override
			public int getGrid0(Integer dataLength) {
				return 1;
			}

			@Override
			public int getBlock0(Integer dataLength) {
				return 1024;
			}

			@Override
			public KernelArg[] createArgs(Integer dataLength) {
				int blocks = Math.min((dataLength + 512 - 1) / 512, 1024);
				long[] in = new long[blocks];
				for (int i = 0; i < in.length; i++) {
					in[i] = i;
				}

				LongArg inArg = LongArg.create(in);
				LongArg outArg = LongArg.createOutput(blocks);
				KernelArg nArg = IntArg.createValue(blocks);

				return new KernelArg[] { inArg, outArg, nArg };
			}
		};

		Integer[] dataSizes = new Integer[] {1024, 2048, 4096, 8192, 256 * 1024*1024};

		// Options
		Options options = Options.createOptions();

		// Warm up
		Executor.benchmark("device_reduce", options, 30, creator1, 1024*1024);

		// Benchmark Reduce-Kernel
		Executor.BenchmarkResult<Integer> result = Executor.benchmark("device_reduce", options, 50, creator1, dataSizes);

		// Simulate second kernel call
		Executor.BenchmarkResult<Integer> result2 = Executor.benchmark("device_reduce", options, 50, creator2, dataSizes);

		// Add the average times of the second benchmark to the first
		Executor.BenchmarkResult<Integer> sum = result.addBenchmarkResult(result2);

		// Print out the final benchmark result
		System.out.println(sum);
	}
}