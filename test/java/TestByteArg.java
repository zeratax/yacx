import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

class TestByteArg extends TestJNI {
	static Kernel cpByteArray, cpByteSingle;
	static int n;
	static byte[] testArray0, testArray1;
	ByteArg inArg, outArg;
	
	@BeforeAll
	static void init() {
		//test-data
		n = 17;
		testArray0 = new byte[n];
		testArray1 = new byte[n];
		
		for (int i = 0; i < n; i++) {
			testArray0[i] = (byte) i;
			testArray1[i] = (byte) (13 + (n-i));
		}
		
		//Kernel for copy byte* in to byte* out
		String copyByteArrayString = "extern \"C\" __global__\n" + 
				"void copyByte(char* in, char* out) {\n" + 
				"  int i = (blockIdx.x * blockDim.x) + threadIdx.x;\n" + 
				"  out[i] = in[i];\n" + 
				"}\n" + 
				"";
		
		cpByteArray = Program.create(copyByteArrayString, "copyByte").compile();
		//Configure with kernel n Threads
		cpByteArray.configure(n, 1);
		
		//Kernel for copy a byte-value
		String copyByteString = "extern \"C\" __global__\n" + 
				"void copyByte(char in, char* out) {\n" + 
				"  *out = in;\n" + 
				"}\n" + 
				"";
		
		cpByteSingle = Program.create(copyByteString, "copyByte").compile();
		//Configure Kernel with 1 thread
		cpByteSingle.configure(1, 1);
	}
	
	/**
	 * Check if testArray is expected testArray0 
	 * @param testArray the array to be tested
	 */
	void checkTestArray0(byte[] testArray) {
		assertEquals(n, testArray.length);
		
		for (int i = 0; i < n; i++) {
			assertEquals((byte) i, testArray[i]);
		}
	}
	
	/**
	 * Check if testArray is expected testArray1 
	 * @param testArray the array to be tested
	 */
	void checkTestArray1(byte[] testArray) {
		assertEquals(n, testArray.length);
		
		for (int i = 0; i < n; i++) {
			assertEquals((byte) (13 + (n-i)), testArray[i]);
		}
	}

	@Test
	void testInvalidParameter() {
		//Check if parameter is null
		assertThrows(NullPointerException.class, () -> {
			ByteArg.create((byte[]) null);
		});
		
		//Check without any parameters
		assertThrows(IllegalArgumentException.class, () -> {
			ByteArg.create();
		});
		
		assertThrows(IllegalArgumentException.class, () -> {
			ByteArg.create(new byte[0]);
		});
		
		//Check create output-array with invalid size
		assertThrows(IllegalArgumentException.class, () -> {
			ByteArg.createOutput(0);
		});
		
		assertThrows(IllegalArgumentException.class, () -> {
			ByteArg.createOutput(-1);
		});
		
		assertThrows(IllegalArgumentException.class, () -> {
			ByteArg.createOutput(Integer.MIN_VALUE);
		});
	}
	
	@Test
	void testByteSingle() {
		KernelArg inArg;
		
		//Create KernelArgs
		inArg = ByteArg.createValue((byte) 4);
		outArg = ByteArg.createOutput(1);
		
		cpByteSingle.launch(inArg, outArg);
		
		//Check result
		assertEquals(1, outArg.asByteArray().length);
		assertEquals(4, outArg.asByteArray()[0]);
		
		//Create KernelArgs
		inArg = ByteArg.createValue((byte) -128);
				
		cpByteSingle.launch(inArg, outArg);
			
		//Check result
		assertEquals(1, outArg.asByteArray().length);
		assertEquals((byte) -128, outArg.asByteArray()[0]);
	}
	
	@Test
	void testByteArray() {
		//Check test-Arrays should be correct
		checkTestArray0(testArray0);
		checkTestArray1(testArray1);
		
		//Create KernelArgs (download both)
		inArg = ByteArg.create(testArray0, true);
		outArg = ByteArg.create(testArray1, true);
		
		cpByteArray.launch(inArg, outArg);
		
		//Check Result
		checkTestArray0(inArg.asByteArray());
		checkTestArray0(outArg.asByteArray());
		//Other Array should be unchanged
		checkTestArray0(testArray0);
		checkTestArray1(testArray1);
		
		
		//Create KernelArgs (download only inArg)
		inArg = ByteArg.create(testArray0, true);
		outArg = ByteArg.create(testArray1, false);
		
		cpByteArray.launch(inArg, outArg);
		
		//Check Result
		checkTestArray0(inArg.asByteArray());
		checkTestArray1(outArg.asByteArray());
		//Other Array should be unchanged
		checkTestArray0(testArray0);
		checkTestArray1(testArray1);
		
		
		//Create KernelArgs (download only outArg)
		inArg = ByteArg.create(testArray0, false);
		outArg = ByteArg.create(testArray1, true);
		
		cpByteArray.launch(inArg, outArg);
		
		//Check Result
		checkTestArray0(inArg.asByteArray());
		checkTestArray0(outArg.asByteArray());
		//Other Array should be unchanged
		checkTestArray0(testArray0);
		checkTestArray1(testArray1);
		
		
		//Create KernelArgs (download nothing)
		inArg = ByteArg.create(testArray0, false);
		outArg = ByteArg.create(testArray1, false);
		
		cpByteArray.launch(inArg, outArg);
		
		//Check Result
		checkTestArray0(inArg.asByteArray());
		checkTestArray1(outArg.asByteArray());
		//Other Array should be unchanged
		checkTestArray0(testArray0);
		checkTestArray1(testArray1);
	}
	
	@Test
	void testByteOutput() {
		//Check test-Arrays should be correct
		checkTestArray0(testArray0);
		checkTestArray1(testArray1);
		
		//Create KernelArgs
		inArg = ByteArg.create(testArray0);
		outArg = ByteArg.createOutput(n);
		
		cpByteArray.launch(inArg, outArg);
		
		//Check Result
		checkTestArray0(inArg.asByteArray());
		checkTestArray0(outArg.asByteArray());
		//Other Array should be unchanged
		checkTestArray0(testArray0);
		checkTestArray1(testArray1);
		
		//Create KernelArgs
		inArg = ByteArg.create(testArray1);
		outArg = ByteArg.createOutput(n);
		
		cpByteArray.launch(inArg, outArg);
		
		//Check Result
		checkTestArray1(inArg.asByteArray());
		checkTestArray1(outArg.asByteArray());
		//Other Array should be unchanged
		checkTestArray0(testArray0);
		checkTestArray1(testArray1);
		
		//Use the same KernelArgs again
		cpByteArray.launch(inArg, outArg);
		
		//Check Result
		checkTestArray1(inArg.asByteArray());
		checkTestArray1(outArg.asByteArray());
		//Other Array should be unchanged
		checkTestArray0(testArray0);
		checkTestArray1(testArray1);
	}
}
