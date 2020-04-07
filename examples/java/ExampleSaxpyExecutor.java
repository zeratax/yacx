import java.io.IOException;
import java.util.Arrays;

import yacx.Executor;
import yacx.FloatArg;
import yacx.IntArg;

public class ExampleSaxpyExecutor {
	public static void main(String[] args) throws IOException {
		// Load Libary
		Executor.loadLibrary();

		// Create OutputArgument
		int n = 4;
		FloatArg out = FloatArg.createOutput(n);

		// Compile and launch Kernel
		System.out.println("\n" + Executor.launch("saxpy", n, 1, FloatArg.createValue(5.1f),
				FloatArg.create(0, 1, 2, 3), FloatArg.create(2, 2, 4, 4), out, IntArg.createValue(n)));

		// Print Result
		System.out.println("Result: " + Arrays.toString(out.asFloatArray()));
	}
}