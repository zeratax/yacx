import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

class TestIntArg extends TestJNI {
	static Kernel cpIntArray, cpIntSingle;
	static int n;
	static int[] testArray0, testArray1;
	IntArg inArg, outArg;
	
	@BeforeAll
	static void init() {
		//test-data
		n = 17;
		testArray0 = new int[n];
		testArray1 = new int[n];
		
		for (int i = 0; i < n; i++) {
			testArray0[i] = (int) i;
			testArray1[i] = (int) (13 + (n-i));
		}
		
		//Kernel for copy int* in to int* out
		String copyIntArrayString = "extern \"C\" __global__\n" + 
				"void copyInt(int* in, int* out) {\n" + 
				"  int i = (blockIdx.x * blockDim.x) + threadIdx.x;\n" + 
				"  out[i] = in[i];\n" + 
				"}\n" + 
				"";
		
		cpIntArray = Program.create(copyIntArrayString, "copyInt").compile();
		//Configure with kernel n Threads
		cpIntArray.configure(n, 1);
		
		//Kernel for copy a int-value
		String copyIntString = "extern \"C\" __global__\n" + 
				"void copyInt(int in, int* out) {\n" + 
				"  *out = in;\n" + 
				"}\n" + 
				"";
		
		cpIntSingle = Program.create(copyIntString, "copyInt").compile();
		//Configure Kernel with 1 thread
		cpIntSingle.configure(1, 1);
	}
	
	/**
	 * Check if testArray is expected testArray0 
	 * @param testArray the array to be tested
	 */
	void checkTestArray0(int[] testArray) {
		assertEquals(n, testArray.length);
		
		for (int i = 0; i < n; i++) {
			assertEquals((int) i, testArray[i]);
		}
	}
	
	/**
	 * Check if testArray is expected testArray1 
	 * @param testArray the array to be tested
	 */
	void checkTestArray1(int[] testArray) {
		assertEquals(n, testArray.length);
		
		for (int i = 0; i < n; i++) {
			assertEquals((int) (13 + (n-i)), testArray[i]);
		}
	}

	@Test
	void testInvalidParameter() {
		//Check if parameter is null
		assertThrows(NullPointerException.class, () -> {
			IntArg.create((int[]) null);
		});
		
		//Check without any parameters
		assertThrows(IllegalArgumentException.class, () -> {
			IntArg.create();
		});
		
		assertThrows(IllegalArgumentException.class, () -> {
			IntArg.create(new int[0]);
		});
		
		//Check create output-array with invalid size
		assertThrows(IllegalArgumentException.class, () -> {
			IntArg.createOutput(0);
		});
		
		assertThrows(IllegalArgumentException.class, () -> {
			IntArg.createOutput(-1);
		});
		
		assertThrows(IllegalArgumentException.class, () -> {
			IntArg.createOutput(Integer.MIN_VALUE);
		});
	}
	
	@Test
	void testIntSingle() {
		KernelArg inArg;
		
		//Create KernelArgs
		inArg = IntArg.createValue((int) 4);
		outArg = IntArg.createOutput(1);
		
		cpIntSingle.launch(inArg, outArg);
		
		//Check result
		assertEquals(1, outArg.asIntArray().length);
		assertEquals(4, outArg.asIntArray()[0]);
		
		//Create KernelArgs
		inArg = IntArg.createValue((int) -128);
				
		cpIntSingle.launch(inArg, outArg);
			
		//Check result
		assertEquals(1, outArg.asIntArray().length);
		assertEquals((int) -128, outArg.asIntArray()[0]);
	}
	
	@Test
	void testIntArray() {
		//Check test-Arrays should be correct
		checkTestArray0(testArray0);
		checkTestArray1(testArray1);
		
		//Create KernelArgs (download both)
		inArg = IntArg.create(testArray0, true);
		outArg = IntArg.create(testArray1, true);
		
		cpIntArray.launch(inArg, outArg);
		
		//Check Result
		checkTestArray0(inArg.asIntArray());
		checkTestArray0(outArg.asIntArray());
		//Other Array should be unchanged
		checkTestArray0(testArray0);
		checkTestArray1(testArray1);
		
		
		//Create KernelArgs (download only inArg)
		inArg = IntArg.create(testArray0, true);
		outArg = IntArg.create(testArray1, false);
		
		cpIntArray.launch(inArg, outArg);
		
		//Check Result
		checkTestArray0(inArg.asIntArray());
		checkTestArray1(outArg.asIntArray());
		//Other Array should be unchanged
		checkTestArray0(testArray0);
		checkTestArray1(testArray1);
		
		
		//Create KernelArgs (download only outArg)
		inArg = IntArg.create(testArray0, false);
		outArg = IntArg.create(testArray1, true);
		
		cpIntArray.launch(inArg, outArg);
		
		//Check Result
		checkTestArray0(inArg.asIntArray());
		checkTestArray0(outArg.asIntArray());
		//Other Array should be unchanged
		checkTestArray0(testArray0);
		checkTestArray1(testArray1);
		
		
		//Create KernelArgs (download nothing)
		inArg = IntArg.create(testArray0, false);
		outArg = IntArg.create(testArray1, false);
		
		cpIntArray.launch(inArg, outArg);
		
		//Check Result
		checkTestArray0(inArg.asIntArray());
		checkTestArray1(outArg.asIntArray());
		//Other Array should be unchanged
		checkTestArray0(testArray0);
		checkTestArray1(testArray1);
	}
	
	@Test
	void testIntOutput() {
		//Check test-Arrays should be correct
		checkTestArray0(testArray0);
		checkTestArray1(testArray1);
		
		//Create KernelArgs
		inArg = IntArg.create(testArray0);
		outArg = IntArg.createOutput(n);
		
		cpIntArray.launch(inArg, outArg);
		
		//Check Result
		checkTestArray0(inArg.asIntArray());
		checkTestArray0(outArg.asIntArray());
		//Other Array should be unchanged
		checkTestArray0(testArray0);
		checkTestArray1(testArray1);
		
		//Create KernelArgs
		inArg = IntArg.create(testArray1);
		outArg = IntArg.createOutput(n);
		
		cpIntArray.launch(inArg, outArg);
		
		//Check Result
		checkTestArray1(inArg.asIntArray());
		checkTestArray1(outArg.asIntArray());
		//Other Array should be unchanged
		checkTestArray0(testArray0);
		checkTestArray1(testArray1);
		
		//Use the same KernelArgs again
		cpIntArray.launch(inArg, outArg);
		
		//Check Result
		checkTestArray1(inArg.asIntArray());
		checkTestArray1(outArg.asIntArray());
		//Other Array should be unchanged
		checkTestArray0(testArray0);
		checkTestArray1(testArray1);
	}
}
