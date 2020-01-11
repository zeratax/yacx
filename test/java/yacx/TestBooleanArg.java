package yacx;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

class TestBooleanArg extends TestJNI {
	static Kernel cpBooleanArray, cpBooleanSingle;
	static int n;
	static boolean[] testArray0, testArray1;
	BooleanArg inArg, outArg;
	
	@BeforeAll
	static void init() {
		//test-data
		n = 17;
		testArray0 = new boolean[n];
		testArray1 = new boolean[n];
		
		for (int i = 0; i < n; i++) {
			testArray0[i] = (i % 2 == 0);
			testArray1[i] = (i != 13);
		}
		
		//Kernel for copy bool* in to bool* out
		String copyBooleanArrayString = "extern \"C\" __global__\n" + 
				"void copyBoolean(bool* in, bool* out) {\n" + 
				"  int i = (blockIdx.x * blockDim.x) + threadIdx.x;\n" + 
				"  out[i] = in[i];\n" + 
				"}\n" + 
				"";
		
		cpBooleanArray = Program.create(copyBooleanArrayString, "copyBoolean").compile();
		//Configure with kernel n Threads
		cpBooleanArray.configure(n, 1);
		
		//Kernel for copy a boolean-value
		String copyBooleanString = "extern \"C\" __global__\n" + 
				"void copyBoolean(bool in, bool* out) {\n" + 
				"  *out = in;\n" + 
				"}\n" + 
				"";
		
		cpBooleanSingle = Program.create(copyBooleanString, "copyBoolean").compile();
		//Configure Kernel with 1 thread
		cpBooleanSingle.configure(1, 1);
	}
	
	/**
	 * Check if testArray is expected testArray0 
	 * @param testArray the array to be tested
	 */
	void checkTestArray0(boolean[] testArray) {
		assertEquals(n, testArray.length);
		
		for (int i = 0; i < n; i++) {
			if (i % 2 == 0)
				assertTrue(testArray[i]);
			else
				assertFalse(testArray[i]);
		}
	}
	
	/**
	 * Check if testArray is expected testArray1 
	 * @param testArray the array to be tested
	 */
	void checkTestArray1(boolean[] testArray) {
		assertEquals(n, testArray.length);
		
		for (int i = 0; i < n; i++) {
			if (i != 13)
				assertTrue(testArray[i]);
			else
				assertFalse(testArray[i]);
		}
	}

	@Test
	void testInvalidParameter() {
		//Check if parameter is null
		assertThrows(NullPointerException.class, () -> {
			BooleanArg.create((boolean[]) null);
		});
		
		//Check without any parameters
		assertThrows(IllegalArgumentException.class, () -> {
			BooleanArg.create();
		});
		
		assertThrows(IllegalArgumentException.class, () -> {
			BooleanArg.create(new boolean[0]);
		});
		
		//Check create output-array with invalid size
		assertThrows(IllegalArgumentException.class, () -> {
			BooleanArg.createOutput(0);
		});
		
		assertThrows(IllegalArgumentException.class, () -> {
			BooleanArg.createOutput(-1);
		});
		
		assertThrows(IllegalArgumentException.class, () -> {
			BooleanArg.createOutput(Integer.MIN_VALUE);
		});
	}
	
	@Test
	void testBooleanSingle() {
		KernelArg inArg;
		
		//Create KernelArgs
		inArg = BooleanArg.createValue(true);
		outArg = BooleanArg.createOutput(1);
		
		cpBooleanSingle.launch(inArg, outArg);
		
		//Check result
		assertEquals(1, outArg.asBooleanArray().length);
		assertTrue(outArg.asBooleanArray()[0]);
		
		//Create KernelArgs
		inArg = BooleanArg.createValue(false);
				
		cpBooleanSingle.launch(inArg, outArg);
			
		//Check result
		assertEquals(1, outArg.asBooleanArray().length);
		assertFalse(outArg.asBooleanArray()[0]);
	}
	
	@Test
	void testBooleanArray() {
		//Check test-Arrays should be correct
		checkTestArray0(testArray0);
		checkTestArray1(testArray1);
		
		//Create KernelArgs (download both)
		inArg = BooleanArg.create(testArray0, true);
		outArg = BooleanArg.create(testArray1, true);
		
		cpBooleanArray.launch(inArg, outArg);
		
		//Check Result
		checkTestArray0(inArg.asBooleanArray());
		checkTestArray0(outArg.asBooleanArray());
		//Other Array should be unchanged
		checkTestArray0(testArray0);
		checkTestArray1(testArray1);
		
		
		//Create KernelArgs (download only inArg)
		inArg = BooleanArg.create(testArray0, true);
		outArg = BooleanArg.create(testArray1, false);
		
		cpBooleanArray.launch(inArg, outArg);
		
		//Check Result
		checkTestArray0(inArg.asBooleanArray());
		checkTestArray1(outArg.asBooleanArray());
		//Other Array should be unchanged
		checkTestArray0(testArray0);
		checkTestArray1(testArray1);
		
		
		//Create KernelArgs (download only outArg)
		inArg = BooleanArg.create(testArray0, false);
		outArg = BooleanArg.create(testArray1, true);
		
		cpBooleanArray.launch(inArg, outArg);
		
		//Check Result
		checkTestArray0(inArg.asBooleanArray());
		checkTestArray0(outArg.asBooleanArray());
		//Other Array should be unchanged
		checkTestArray0(testArray0);
		checkTestArray1(testArray1);
		
		
		//Create KernelArgs (download nothing)
		inArg = BooleanArg.create(testArray0, false);
		outArg = BooleanArg.create(testArray1, false);
		
		cpBooleanArray.launch(inArg, outArg);
		
		//Check Result
		checkTestArray0(inArg.asBooleanArray());
		checkTestArray1(outArg.asBooleanArray());
		//Other Array should be unchanged
		checkTestArray0(testArray0);
		checkTestArray1(testArray1);
	}
	
	@Test
	void testBooleanOutput() {
		//Check test-Arrays should be correct
		checkTestArray0(testArray0);
		checkTestArray1(testArray1);
		
		//Create KernelArgs
		inArg = BooleanArg.create(testArray0);
		outArg = BooleanArg.createOutput(n);
		
		cpBooleanArray.launch(inArg, outArg);
		
		//Check Result
		checkTestArray0(inArg.asBooleanArray());
		checkTestArray0(outArg.asBooleanArray());
		//Other Array should be unchanged
		checkTestArray0(testArray0);
		checkTestArray1(testArray1);
		
		//Create KernelArgs
		inArg = BooleanArg.create(testArray1);
		outArg = BooleanArg.createOutput(n);
		
		cpBooleanArray.launch(inArg, outArg);
		
		//Check Result
		checkTestArray1(inArg.asBooleanArray());
		checkTestArray1(outArg.asBooleanArray());
		//Other Array should be unchanged
		checkTestArray0(testArray0);
		checkTestArray1(testArray1);
		
		//Use the same KernelArgs again
		cpBooleanArray.launch(inArg, outArg);
		
		//Check Result
		checkTestArray1(inArg.asBooleanArray());
		checkTestArray1(outArg.asBooleanArray());
		//Other Array should be unchanged
		checkTestArray0(testArray0);
		checkTestArray1(testArray1);
	}
}
