import java.io.IOException;
import java.util.Arrays;

import yacx.Executor;
import yacx.IntArg;
import yacx.Kernel;
import yacx.KernelArg;
import yacx.LongArg;
import yacx.Options;
import yacx.Program;
import yacx.Utils;

public class ExampleReduce {
	public static void main(String[] args) throws IOException {
		// Load Libary
		Executor.loadLibary();

		// Testdata
		int arraySize = 26251;

		final int numBlocks = 512;
		final int numThreads = Math.min((arraySize + numBlocks - 1) / numBlocks, 1024);

		long[] in = new long[arraySize];
		for (int i = 1; i <= in.length; i++) {
			in[i - 1] = i;
		}

		// Initialize Arguments
		LongArg inArg = LongArg.create(in, false);
		LongArg outArg = LongArg.createOutput(arraySize);
		KernelArg nArg = IntArg.createValue(arraySize);

		// Load kernelString
		String kernelString = Utils.loadFile("kernels/device_reduce.cu");
		// Create Program
		Program deviceReduce = Program.create(kernelString, "deviceReduceKernel");

		// Options for using C++14
		Options options = Options.createOptions("--std=c++14");

		// Create compiled Kernel
		Kernel deviceReduceKernel = deviceReduce.compile(options);

		// Launch Kernel
		deviceReduceKernel.launch(numThreads, numBlocks, inArg, outArg, nArg);

		// New Input is Output from previous run
		inArg = LongArg.create(outArg.asLongArray());
		// Second launch
		deviceReduceKernel.launch(numThreads, numBlocks, inArg, outArg, nArg);

		// Get Result
		long out = outArg.asLongArray()[0];

		// Print Result
		System.out.println("\nInput:");
		System.out.println(Arrays.toString(in));
		System.out.println("\nResult:   " + out);
		System.out.println("Expected: " + (arraySize * (arraySize + 1)) / 2);
	}
}
