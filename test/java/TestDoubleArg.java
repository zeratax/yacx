import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

class TestDoubleArg extends TestJNI {
	static Kernel cpDoubleArray, cpDoubleSingle;
	static int n;
	static double[] testArray0, testArray1;
	DoubleArg inArg, outArg;
	
	@BeforeAll
	static void init() {
		//test-data
		n = 17;
		testArray0 = new double[n];
		testArray1 = new double[n];
		
		for (int i = 0; i < n; i++) {
			testArray0[i] = 1.1d * i;
			testArray1[i] = 13.3d + (n-i) * 0.999d;
		}
		
		//Kernel for copy double* in to double* out
		String copyDoubleArrayString = "extern \"C\" __global__\n" + 
				"void copyDouble(double* in, double* out) {\n" + 
				"  int i = (blockIdx.x * blockDim.x) + threadIdx.x;\n" + 
				"  out[i] = in[i];\n" + 
				"}\n" + 
				"";
		
		cpDoubleArray = Program.create(copyDoubleArrayString, "copyDouble").compile();
		//Configure with kernel n Threads
		cpDoubleArray.configure(n, 1);
		
		//Kernel for copy a double-value
		String copyDoubleString = "extern \"C\" __global__\n" + 
				"void copyDouble(double in, double* out) {\n" + 
				"  *out = in;\n" + 
				"}\n" + 
				"";
		
		cpDoubleSingle = Program.create(copyDoubleString, "copyDouble").compile();
		//Configure Kernel with 1 thread
		cpDoubleSingle.configure(1, 1);
	}
	
	/**
	 * Check if testArray is expected testArray0 
	 * @param testArray the array to be tested
	 */
	void checkTestArray0(double[] testArray) {
		assertEquals(n, testArray.length);
		
		for (int i = 0; i < n; i++) {
			assertEquals(1.1d * i, testArray[i]);
		}
	}
	
	/**
	 * Check if testArray is expected testArray1 
	 * @param testArray the array to be tested
	 */
	void checkTestArray1(double[] testArray) {
		assertEquals(n, testArray.length);
		
		for (int i = 0; i < n; i++) {
			assertEquals(13.3d + (n-i) * 0.999d, testArray[i]);
		}
	}

	@Test
	void testInvalidParameter() {
		//Check if parameter is null
		assertThrows(NullPointerException.class, () -> {
			DoubleArg.create((double[]) null);
		});
		
		//Check without any parameters
		assertThrows(IllegalArgumentException.class, () -> {
			DoubleArg.create();
		});
		
		assertThrows(IllegalArgumentException.class, () -> {
			DoubleArg.create(new double[0]);
		});
		
		//Check create output-array with invalid size
		assertThrows(IllegalArgumentException.class, () -> {
			DoubleArg.createOutput(0);
		});
		
		assertThrows(IllegalArgumentException.class, () -> {
			DoubleArg.createOutput(-1);
		});
		
		assertThrows(IllegalArgumentException.class, () -> {
			DoubleArg.createOutput(Integer.MIN_VALUE);
		});
	}
	
	@Test
	void testDoubleSingle() {
		KernelArg inArg;
		
		//Create KernelArgs
		inArg = DoubleArg.createValue(4.9d);
		outArg = DoubleArg.createOutput(1);
		
		cpDoubleSingle.launch(inArg, outArg);
		
		//Check result
		assertEquals(1, outArg.asDoubleArray().length);
		assertEquals(4.9d, outArg.asDoubleArray()[0]);
		
		//Create KernelArgs
		inArg = DoubleArg.createValue(-128.1d);
				
		cpDoubleSingle.launch(inArg, outArg);
			
		//Check result
		assertEquals(1, outArg.asDoubleArray().length);
		assertEquals(-128.1d, outArg.asDoubleArray()[0]);
	}
	
	@Test
	void testDoubleArray() {
		//Check test-Arrays should be correct
		checkTestArray0(testArray0);
		checkTestArray1(testArray1);
		
		//Create KernelArgs (download both)
		inArg = DoubleArg.create(testArray0, true);
		outArg = DoubleArg.create(testArray1, true);
		
		cpDoubleArray.launch(inArg, outArg);
		
		//Check Result
		checkTestArray0(inArg.asDoubleArray());
		checkTestArray0(outArg.asDoubleArray());
		//Other Array should be unchanged
		checkTestArray0(testArray0);
		checkTestArray1(testArray1);
		
		
		//Create KernelArgs (download only inArg)
		inArg = DoubleArg.create(testArray0, true);
		outArg = DoubleArg.create(testArray1, false);
		
		cpDoubleArray.launch(inArg, outArg);
		
		//Check Result
		checkTestArray0(inArg.asDoubleArray());
		checkTestArray1(outArg.asDoubleArray());
		//Other Array should be unchanged
		checkTestArray0(testArray0);
		checkTestArray1(testArray1);
		
		
		//Create KernelArgs (download only outArg)
		inArg = DoubleArg.create(testArray0, false);
		outArg = DoubleArg.create(testArray1, true);
		
		cpDoubleArray.launch(inArg, outArg);
		
		//Check Result
		checkTestArray0(inArg.asDoubleArray());
		checkTestArray0(outArg.asDoubleArray());
		//Other Array should be unchanged
		checkTestArray0(testArray0);
		checkTestArray1(testArray1);
		
		
		//Create KernelArgs (download nothing)
		inArg = DoubleArg.create(testArray0, false);
		outArg = DoubleArg.create(testArray1, false);
		
		cpDoubleArray.launch(inArg, outArg);
		
		//Check Result
		checkTestArray0(inArg.asDoubleArray());
		checkTestArray1(outArg.asDoubleArray());
		//Other Array should be unchanged
		checkTestArray0(testArray0);
		checkTestArray1(testArray1);
	}
	
	@Test
	void testDoubleOutput() {
		//Check test-Arrays should be correct
		checkTestArray0(testArray0);
		checkTestArray1(testArray1);
		
		//Create KernelArgs
		inArg = DoubleArg.create(testArray0);
		outArg = DoubleArg.createOutput(n);
		
		cpDoubleArray.launch(inArg, outArg);
		
		//Check Result
		checkTestArray0(inArg.asDoubleArray());
		checkTestArray0(outArg.asDoubleArray());
		//Other Array should be unchanged
		checkTestArray0(testArray0);
		checkTestArray1(testArray1);
		
		//Create KernelArgs
		inArg = DoubleArg.create(testArray1);
		outArg = DoubleArg.createOutput(n);
		
		cpDoubleArray.launch(inArg, outArg);
		
		//Check Result
		checkTestArray1(inArg.asDoubleArray());
		checkTestArray1(outArg.asDoubleArray());
		//Other Array should be unchanged
		checkTestArray0(testArray0);
		checkTestArray1(testArray1);
		
		//Use the same KernelArgs again
		cpDoubleArray.launch(inArg, outArg);
		
		//Check Result
		checkTestArray1(inArg.asDoubleArray());
		checkTestArray1(outArg.asDoubleArray());
		//Other Array should be unchanged
		checkTestArray0(testArray0);
		checkTestArray1(testArray1);
	}
}
