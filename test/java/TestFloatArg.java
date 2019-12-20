package yacx;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

class TestFloatArg extends TestJNI {
	static Kernel cpFloatArray, cpFloatSingle;
	static int n;
	static float[] testArray0, testArray1;
	FloatArg inArg, outArg;
	
	@BeforeAll
	static void init() {
		//test-data
		n = 17;
		testArray0 = new float[n];
		testArray1 = new float[n];
		
		for (int i = 0; i < n; i++) {
			testArray0[i] = 1.1f * i;
			testArray1[i] = 13.3f + (n-i) * 0.999f;
		}
		
		//Kernel for copy float* in to float* out
		String copyFloatArrayString = "extern \"C\" __global__\n" + 
				"void copyFloat(float* in, float* out) {\n" + 
				"  int i = (blockIdx.x * blockDim.x) + threadIdx.x;\n" + 
				"  out[i] = in[i];\n" + 
				"}\n" + 
				"";
		
		cpFloatArray = Program.create(copyFloatArrayString, "copyFloat").compile();
		//Configure with kernel n Threads
		cpFloatArray.configure(n, 1);
		
		//Kernel for copy a float-value
		String copyFloatString = "extern \"C\" __global__\n" + 
				"void copyFloat(float in, float* out) {\n" + 
				"  *out = in;\n" + 
				"}\n" + 
				"";
		
		cpFloatSingle = Program.create(copyFloatString, "copyFloat").compile();
		//Configure Kernel with 1 thread
		cpFloatSingle.configure(1, 1);
	}
	
	/**
	 * Check if testArray is expected testArray0 
	 * @param testArray the array to be tested
	 */
	void checkTestArray0(float[] testArray) {
		assertEquals(n, testArray.length);
		
		for (int i = 0; i < n; i++) {
			assertEquals(1.1f * i, testArray[i]);
		}
	}
	
	/**
	 * Check if testArray is expected testArray1 
	 * @param testArray the array to be tested
	 */
	void checkTestArray1(float[] testArray) {
		assertEquals(n, testArray.length);
		
		for (int i = 0; i < n; i++) {
			assertEquals(13.3f + (n-i) * 0.999f, testArray[i]);
		}
	}

	@Test
	void testInvalidParameter() {
		//Check if parameter is null
		assertThrows(NullPointerException.class, () -> {
			FloatArg.create((float[]) null);
		});
		
		//Check without any parameters
		assertThrows(IllegalArgumentException.class, () -> {
			FloatArg.create();
		});
		
		assertThrows(IllegalArgumentException.class, () -> {
			FloatArg.create(new float[0]);
		});
		
		//Check create output-array with invalid size
		assertThrows(IllegalArgumentException.class, () -> {
			FloatArg.createOutput(0);
		});
		
		assertThrows(IllegalArgumentException.class, () -> {
			FloatArg.createOutput(-1);
		});
		
		assertThrows(IllegalArgumentException.class, () -> {
			FloatArg.createOutput(Integer.MIN_VALUE);
		});
	}
	
	@Test
	void testFloatSingle() {
		KernelArg inArg;
		
		//Create KernelArgs
		inArg = FloatArg.createValue(4.9f);
		outArg = FloatArg.createOutput(1);
		
		cpFloatSingle.launch(inArg, outArg);
		
		//Check result
		assertEquals(1, outArg.asFloatArray().length);
		assertEquals(4.9f, outArg.asFloatArray()[0]);
		
		//Create KernelArgs
		inArg = FloatArg.createValue(-128.1f);
				
		cpFloatSingle.launch(inArg, outArg);
			
		//Check result
		assertEquals(1, outArg.asFloatArray().length);
		assertEquals(-128.1f, outArg.asFloatArray()[0]);
	}
	
	@Test
	void testFloatArray() {
		//Check test-Arrays should be correct
		checkTestArray0(testArray0);
		checkTestArray1(testArray1);
		
		//Create KernelArgs (download both)
		inArg = FloatArg.create(testArray0, true);
		outArg = FloatArg.create(testArray1, true);
		
		cpFloatArray.launch(inArg, outArg);
		
		//Check Result
		checkTestArray0(inArg.asFloatArray());
		checkTestArray0(outArg.asFloatArray());
		//Other Array should be unchanged
		checkTestArray0(testArray0);
		checkTestArray1(testArray1);
		
		
		//Create KernelArgs (download only inArg)
		inArg = FloatArg.create(testArray0, true);
		outArg = FloatArg.create(testArray1, false);
		
		cpFloatArray.launch(inArg, outArg);
		
		//Check Result
		checkTestArray0(inArg.asFloatArray());
		checkTestArray1(outArg.asFloatArray());
		//Other Array should be unchanged
		checkTestArray0(testArray0);
		checkTestArray1(testArray1);
		
		
		//Create KernelArgs (download only outArg)
		inArg = FloatArg.create(testArray0, false);
		outArg = FloatArg.create(testArray1, true);
		
		cpFloatArray.launch(inArg, outArg);
		
		//Check Result
		checkTestArray0(inArg.asFloatArray());
		checkTestArray0(outArg.asFloatArray());
		//Other Array should be unchanged
		checkTestArray0(testArray0);
		checkTestArray1(testArray1);
		
		
		//Create KernelArgs (download nothing)
		inArg = FloatArg.create(testArray0, false);
		outArg = FloatArg.create(testArray1, false);
		
		cpFloatArray.launch(inArg, outArg);
		
		//Check Result
		checkTestArray0(inArg.asFloatArray());
		checkTestArray1(outArg.asFloatArray());
		//Other Array should be unchanged
		checkTestArray0(testArray0);
		checkTestArray1(testArray1);
	}
	
	@Test
	void testFloatOutput() {
		//Check test-Arrays should be correct
		checkTestArray0(testArray0);
		checkTestArray1(testArray1);
		
		//Create KernelArgs
		inArg = FloatArg.create(testArray0);
		outArg = FloatArg.createOutput(n);
		
		cpFloatArray.launch(inArg, outArg);
		
		//Check Result
		checkTestArray0(inArg.asFloatArray());
		checkTestArray0(outArg.asFloatArray());
		//Other Array should be unchanged
		checkTestArray0(testArray0);
		checkTestArray1(testArray1);
		
		//Create KernelArgs
		inArg = FloatArg.create(testArray1);
		outArg = FloatArg.createOutput(n);
		
		cpFloatArray.launch(inArg, outArg);
		
		//Check Result
		checkTestArray1(inArg.asFloatArray());
		checkTestArray1(outArg.asFloatArray());
		//Other Array should be unchanged
		checkTestArray0(testArray0);
		checkTestArray1(testArray1);
		
		//Use the same KernelArgs again
		cpFloatArray.launch(inArg, outArg);
		
		//Check Result
		checkTestArray1(inArg.asFloatArray());
		checkTestArray1(outArg.asFloatArray());
		//Other Array should be unchanged
		checkTestArray0(testArray0);
		checkTestArray1(testArray1);
	}
}
