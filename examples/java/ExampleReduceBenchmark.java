import java.io.IOException;

import yacx.Executor;
import yacx.Executor.KernelArgCreator;
import yacx.LongArg;
import yacx.IntArg;
import yacx.KernelArg;
import yacx.Options;

public class ExampleReduceBenchmark {
	private final static long KB = 1024;
	private final static long MB = 1024 * 1024;

	public static void main(String[] args) throws IOException {
		// Load Libary
		Executor.loadLibary();

		Options options = Options.createOptions("--gpu-architecture=compute_70");

		KernelArgCreator creator1 = new Executor.KernelArgCreator() {

			@Override
			public int getDataLength(long dataSizeBytes) {
				return (int) (dataSizeBytes / LongArg.SIZE_BYTES);
			}

			@Override
			public int getGrid0(int dataLength) {
				int numThreads = getBlock0(0);
				return Math.min((dataLength + numThreads - 1) / numThreads, 1024);
			}

			@Override
			public int getBlock0(int dataLength) {
				return 512;
			}

			@Override
			public KernelArg[] createArgs(int dataLength) {
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

		KernelArgCreator creator2 = new Executor.KernelArgCreator() {

			@Override
			public int getDataLength(long dataSizeBytes) {
				return (int) (dataSizeBytes / LongArg.SIZE_BYTES);
			}

			@Override
			public int getGrid0(int dataLength) {
				return 1;
			}

			@Override
			public int getBlock0(int dataLength) {
				return 1024;
			}

			@Override
			public KernelArg[] createArgs(int dataLength) {
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

		long[] dataSizes = new long[] { 1 * KB, 4 * KB, 16 * KB, 64 * KB, 256 * KB, 1 * MB, 4 * MB, 16 * MB, 64 * MB,
				256 * MB, 1024 * MB };

		// Warm up
		Executor.benchmark("device_reduce", options, 30, creator1, 256 * MB);

		// Benchmark Reduce-Kernel
		Executor.BenchmarkResult result = Executor.benchmark("device_reduce", options, 50, creator1, dataSizes);

		// Simulate second kernel call
		Executor.BenchmarkResult result2 = Executor.benchmark("device_reduce", options, 50, creator2, dataSizes);

		// Add the average times of the second benchmark to the first
		Executor.BenchmarkResult sum = result.addBenchmarkResult(result2);

		// Print out the final benchmark result
		System.out.println(sum);
	}
}