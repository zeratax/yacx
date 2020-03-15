import java.io.IOException;
import java.util.Arrays;

import yacx.Executor;
import yacx.IntArg;
import yacx.Kernel;
import yacx.KernelArg;
import yacx.KernelTime;
import yacx.LongArg;
import yacx.Options;
import yacx.Program;
import yacx.Utils;

public class ExampleReduce {
	public static void main(String[] args) throws IOException {
		// Load Libary
		Executor.loadLibary();

		// Testdata
		int arraySize = 49631;

		final int numThreads = 512;
		final int numBlocks = Math.min((arraySize + numThreads - 1) / numThreads, 1024);

		long[] in = new long[arraySize];
		for (int i = 0; i < in.length; i++) {
			in[i] = i + 1;
		}

		// Initialize Arguments
		LongArg inArg = LongArg.create(in);
		LongArg outArg = LongArg.createOutput(arraySize);
		KernelArg nArg = IntArg.createValue(arraySize);

		// Load kernelString
		String kernelString = Utils.loadFile("kernels/device_reduce.cu");

		// Options for using C++11
		Options options = Options.createOptions("--std=c++11");
		options.insert("--gpu-architecture=compute_70");
		options.insert("-default-device");


		// Launch Kernel
		KernelTime time = Executor.launch(kernelString, "device_reduce", options, numBlocks, numThreads, inArg, outArg, nArg);

		// New Input is Output from previous run
		inArg = LongArg.create(outArg.asLongArray());
		// Second launch
		time.addKernelTime(Executor.launch(kernelString, "device_reduce", options, 1, 1024, inArg, outArg, nArg));

		// Get Result
		long out = outArg.asLongArray()[0];
		long expected = ((long)arraySize * ((long)arraySize + 1)) / 2;

		// Print Result
		System.out.println("Kernel deviceReduce launched " + time.toString());
		System.out.println();
		System.out.println("\nInput: 1..." + arraySize);
		System.out.println("\nResult:   " + out);
		System.out.println("Expected: " + expected);
	}
}
