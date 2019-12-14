import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

class TestShortArg extends TestJNI {
	static Kernel cpShortArray, cpShortSingle;
	static int n;
	static short[] testArray0, testArray1;
	ShortArg inArg, outArg;
	
	@BeforeAll
	static void init() {
		//test-data
		n = 17;
		testArray0 = new short[n];
		testArray1 = new short[n];
		
		for (int i = 0; i < n; i++) {
			testArray0[i] = (short) i;
			testArray1[i] = (short) (13 + (n-i));
		}
		
		//Kernel for copy short* in to short* out
		String copyShortArrayString = "extern \"C\" __global__\n" + 
				"void copyShort(short* in, short* out) {\n" + 
				"  int i = (blockIdx.x * blockDim.x) + threadIdx.x;\n" + 
				"  out[i] = in[i];\n" + 
				"}\n" + 
				"";
		
		cpShortArray = Program.create(copyShortArrayString, "copyShort").compile();
		//Configure with kernel n Threads
		cpShortArray.configure(n, 1);
		
		//Kernel for copy a short-value
		String copyShortString = "extern \"C\" __global__\n" + 
				"void copyShort(short in, short* out) {\n" + 
				"  *out = in;\n" + 
				"}\n" + 
				"";
		
		cpShortSingle = Program.create(copyShortString, "copyShort").compile();
		//Configure Kernel with 1 thread
		cpShortSingle.configure(1, 1);
	}
	
	/**
	 * Check if testArray is expected testArray0 
	 * @param testArray the array to be tested
	 */
	void checkTestArray0(short[] testArray) {
		assertEquals(n, testArray.length);
		
		for (int i = 0; i < n; i++) {
			assertEquals((short) i, testArray[i]);
		}
	}
	
	/**
	 * Check if testArray is expected testArray1 
	 * @param testArray the array to be tested
	 */
	void checkTestArray1(short[] testArray) {
		assertEquals(n, testArray.length);
		
		for (int i = 0; i < n; i++) {
			assertEquals((short) (13 + (n-i)), testArray[i]);
		}
	}

	@Test
	void testInvalidParameter() {
		//Check if parameter is null
		assertThrows(NullPointerException.class, () -> {
			ShortArg.create((short[]) null);
		});
		
		//Check without any parameters
		assertThrows(IllegalArgumentException.class, () -> {
			ShortArg.create();
		});
		
		assertThrows(IllegalArgumentException.class, () -> {
			ShortArg.create(new short[0]);
		});
		
		//Check create output-array with invalid size
		assertThrows(IllegalArgumentException.class, () -> {
			ShortArg.createOutput(0);
		});
		
		assertThrows(IllegalArgumentException.class, () -> {
			ShortArg.createOutput(-1);
		});
		
		assertThrows(IllegalArgumentException.class, () -> {
			ShortArg.createOutput(Integer.MIN_VALUE);
		});
	}
	
	@Test
	void testShortSingle() {
		KernelArg inArg;
		
		//Create KernelArgs
		inArg = ShortArg.createValue((short) 4);
		outArg = ShortArg.createOutput(1);
		
		cpShortSingle.launch(inArg, outArg);
		
		//Check result
		assertEquals(1, outArg.asShortArray().length);
		assertEquals(4, outArg.asShortArray()[0]);
		
		//Create KernelArgs
		inArg = ShortArg.createValue((short) -128);
				
		cpShortSingle.launch(inArg, outArg);
			
		//Check result
		assertEquals(1, outArg.asShortArray().length);
		assertEquals((short) -128, outArg.asShortArray()[0]);
	}
	
	@Test
	void testShortArray() {
		//Check test-Arrays should be correct
		checkTestArray0(testArray0);
		checkTestArray1(testArray1);
		
		//Create KernelArgs (download both)
		inArg = ShortArg.create(testArray0, true);
		outArg = ShortArg.create(testArray1, true);
		
		cpShortArray.launch(inArg, outArg);
		
		//Check Result
		checkTestArray0(inArg.asShortArray());
		checkTestArray0(outArg.asShortArray());
		//Other Array should be unchanged
		checkTestArray0(testArray0);
		checkTestArray1(testArray1);
		
		
		//Create KernelArgs (download only inArg)
		inArg = ShortArg.create(testArray0, true);
		outArg = ShortArg.create(testArray1, false);
		
		cpShortArray.launch(inArg, outArg);
		
		//Check Result
		checkTestArray0(inArg.asShortArray());
		checkTestArray1(outArg.asShortArray());
		//Other Array should be unchanged
		checkTestArray0(testArray0);
		checkTestArray1(testArray1);
		
		
		//Create KernelArgs (download only outArg)
		inArg = ShortArg.create(testArray0, false);
		outArg = ShortArg.create(testArray1, true);
		
		cpShortArray.launch(inArg, outArg);
		
		//Check Result
		checkTestArray0(inArg.asShortArray());
		checkTestArray0(outArg.asShortArray());
		//Other Array should be unchanged
		checkTestArray0(testArray0);
		checkTestArray1(testArray1);
		
		
		//Create KernelArgs (download nothing)
		inArg = ShortArg.create(testArray0, false);
		outArg = ShortArg.create(testArray1, false);
		
		cpShortArray.launch(inArg, outArg);
		
		//Check Result
		checkTestArray0(inArg.asShortArray());
		checkTestArray1(outArg.asShortArray());
		//Other Array should be unchanged
		checkTestArray0(testArray0);
		checkTestArray1(testArray1);
	}
	
	@Test
	void testShortOutput() {
		//Check test-Arrays should be correct
		checkTestArray0(testArray0);
		checkTestArray1(testArray1);
		
		//Create KernelArgs
		inArg = ShortArg.create(testArray0);
		outArg = ShortArg.createOutput(n);
		
		cpShortArray.launch(inArg, outArg);
		
		//Check Result
		checkTestArray0(inArg.asShortArray());
		checkTestArray0(outArg.asShortArray());
		//Other Array should be unchanged
		checkTestArray0(testArray0);
		checkTestArray1(testArray1);
		
		//Create KernelArgs
		inArg = ShortArg.create(testArray1);
		outArg = ShortArg.createOutput(n);
		
		cpShortArray.launch(inArg, outArg);
		
		//Check Result
		checkTestArray1(inArg.asShortArray());
		checkTestArray1(outArg.asShortArray());
		//Other Array should be unchanged
		checkTestArray0(testArray0);
		checkTestArray1(testArray1);
		
		//Use the same KernelArgs again
		cpShortArray.launch(inArg, outArg);
		
		//Check Result
		checkTestArray1(inArg.asShortArray());
		checkTestArray1(outArg.asShortArray());
		//Other Array should be unchanged
		checkTestArray0(testArray0);
		checkTestArray1(testArray1);
	}
}
