import java.io.IOException;
import java.util.Arrays;

import yacx.Executor;
import yacx.FloatArg;
import yacx.KernelTime;

public class ExampleDotProduct {
	public static void main(String[] args) throws IOException {
		// Load library
		Executor.loadLibary();

		// Testdata
		int numberElements = 9;
		
		float[] x = new float[9];
		float[] y = new float[9];

		for (int i = 0; i < 9; i++) {
			x[i] = i;
			y[i] = 2 * i;
		}

		// Initalize arguments
		FloatArg xArg = FloatArg.create(x);
		FloatArg yArg = FloatArg.create(y);
		FloatArg outArg = FloatArg.createOutput(1);

		// Compile and Launch
		KernelTime executionTime = Executor.launch("dotProduct", 1, numberElements, xArg, yArg, outArg);

		// Get Result
		float result = outArg.asFloatArray()[0];

		// Print Result
		System.out.println("\ndotProduct-Kernel sucessfully launched:");
		System.out.println(executionTime);
		System.out.println("\nInput x: " + Arrays.toString(x));
		System.out.println("Input y: " + Arrays.toString(y));
		System.out.println("Result:  " + result);
	}
}
