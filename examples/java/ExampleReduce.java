import java.io.IOException;

import yacx.Executor;
import yacx.IntArg;
import yacx.KernelArg;
import yacx.KernelTime;
import yacx.LongArg;

public class ExampleReduce {
	public static void main(String[] args) throws IOException {
		// Load library
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

		// Launch Kernel
		KernelTime time = Executor.launch("device_reduce", numBlocks, numThreads, inArg, outArg, nArg);

		// New Input is Output from previous run
		outArg.setUpload(true);
		outArg.setDownload(false);
		// Use Input from previous run as Output
		inArg.setUpload(false);
		inArg.setDownload(true);

		// Second launch
		time.addKernelTime(Executor.launch("device_reduce", 1, 1024, outArg, inArg, nArg));

		// Get Result
		long out = inArg.asLongArray()[0];
		long expected = ((long) arraySize * ((long) arraySize + 1)) / 2;

		// Print Result
		System.out.println("Kernel deviceReduce launched " + time.toString());
		System.out.println();
		System.out.println("\nInput: 1..." + arraySize);
		System.out.println("\nResult:   " + out);
		System.out.println("Expected: " + expected);
	}
}
