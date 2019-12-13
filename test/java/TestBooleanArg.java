import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

class TestBooleanArg extends TestJNI {
	static Kernel cpBoolean;
	BooleanArg inArg, outArg;
	
	@BeforeAll
	static void init() {
		String copyBooleanString = "extern \"C\" __global__\n" + 
				"void copyBoolean(bool* in, bool* out) {\n" + 
				"  int i = (blockIdx.x * blockDim.x) + threadIdx.x;\n" + 
				"  out[i] = in[i];\n" + 
				"}\n" + 
				"";
		
		cpBoolean = Program.create(copyBooleanString, "copyBoolean").compile();
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
	void testSingleBoolean() {
		//TODO
	}

	@Test
	void testBooleanArray() {
		//Test data
		int n = 17;
		boolean[] testArray0 = new boolean[n];
		boolean[] testArray1 = new boolean[n];
		
		for (int i = 0; i < n; i++) {
			testArray0[i] = (i % 2 == 0);
			testArray1[i] = (i != 13);
		}
		
		//Configure with n Threads
		cpBoolean.configure(n, 1);
		
		//Create KernelArgs
		inArg = BooleanArg.create(testArray0, true);
		outArg = BooleanArg.create(testArray1, true);
		
		cpBoolean.launch(inArg, outArg);
		
		//Check Result
		checkTestArray0(inArg.asBooleanArray(), n);
		checkTestArray0(outArg.asBooleanArray(), n);
		//Other Array should be unchanged
		checkTestArray0(testArray0, n);
		checkTestArray1(testArray1, n);
		
		
		//Create KernelArgs
		inArg = BooleanArg.create(testArray0, true);
		outArg = BooleanArg.create(testArray1, false);
		
		cpBoolean.launch(inArg, outArg);
		
		//Check Result
		checkTestArray0(inArg.asBooleanArray(), n);
		checkTestArray1(outArg.asBooleanArray(), n);
		//Other Array should be unchanged
		checkTestArray0(testArray0, n);
		checkTestArray1(testArray1, n);
		
		
		//Create KernelArgs
		inArg = BooleanArg.create(testArray0, false);
		outArg = BooleanArg.create(testArray1, true);
		
		cpBoolean.launch(inArg, outArg);
		
		//Check Result
		checkTestArray0(inArg.asBooleanArray(), n);
		checkTestArray0(outArg.asBooleanArray(), n);
		//Other Array should be unchanged
		checkTestArray0(testArray0, n);
		checkTestArray1(testArray1, n);
		
		
		//Create KernelArgs
		inArg = BooleanArg.create(testArray0, false);
		outArg = BooleanArg.create(testArray1, false);
		
		cpBoolean.launch(inArg, outArg);
		
		//Check Result
		checkTestArray0(inArg.asBooleanArray(), n);
		checkTestArray1(outArg.asBooleanArray(), n);
		//Other Array should be unchanged
		checkTestArray0(testArray0, n);
		checkTestArray1(testArray1, n);
	}
	
	@Test
	void testBooleanOutput() {
		//TODO
	}
	
	void checkTestArray0(boolean[] testArray, int length) {
		assertEquals(length, testArray.length);
		
		for (int i = 0; i < length; i++) {
			if (i % 2 == 0)
				assertTrue(testArray[i]);
			else
				assertFalse(testArray[i]);
		}
	}
	
	void checkTestArray1(boolean[] testArray, int length) {
		assertEquals(length, testArray.length);
		
		for (int i = 0; i < length; i++) {
			if (i != 13)
				assertTrue(testArray[i]);
			else
				assertFalse(testArray[i]);
		}
	}
}
