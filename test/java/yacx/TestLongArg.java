package yacx;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

class TestLongArg extends TestJNI {
	static Kernel cpLongArray, cpLongSingle;
	static int n;
	static long[] testArray0, testArray1;
	LongArg inArg, outArg;

	@BeforeAll
	static void init() {
		// test-data
		n = 17;
		testArray0 = new long[n];
		testArray1 = new long[n];

		for (int i = 0; i < n; i++) {
			testArray0[i] = (long) i;
			testArray1[i] = (long) (13 + (n - i));
		}

		// Kernel for copy long* in to long* out
		String copyLongArrayString = "extern \"C\" __global__\n"
				+ "void copyLong(long* in, long* out) {\n"
				+ "  int i = (blockIdx.x * blockDim.x) + threadIdx.x;\n"
				+ "  out[i] = in[i];\n"
				+ "}\n"
				+ "";

		cpLongArray = Program.create(copyLongArrayString, "copyLong").compile();
		// Configure with kernel n Threads
		cpLongArray.configure(n, 1);

		// Kernel for copy a long-value
		String copyLongString = "extern \"C\" __global__\n" + "void copyLong(long in, long* out) {\n" + "  *out = in;\n"
				+ "}\n" + "";

		cpLongSingle = Program.create(copyLongString, "copyLong").compile();
		// Configure Kernel with 1 thread
		cpLongSingle.configure(1, 1);
	}

	/**
	 * Check if testArray is expected testArray0
	 * 
	 * @param testArray the array to be tested
	 */
	void checkTestArray0(long[] testArray) {
		assertEquals(n, testArray.length);

		for (int i = 0; i < n; i++) {
			assertEquals((long) i, testArray[i]);
		}
	}

	/**
	 * Check if testArray is expected testArray1
	 * 
	 * @param testArray the array to be tested
	 */
	void checkTestArray1(long[] testArray) {
		assertEquals(n, testArray.length);

		for (int i = 0; i < n; i++) {
			assertEquals((long) (13 + (n - i)), testArray[i]);
		}
	}

	@Test
	void testInvalidParameter() {
		// Check if parameter is null
		assertThrows(NullPointerException.class, () -> {
			LongArg.create((long[]) null);
		});

		// Check without any parameters
		assertThrows(IllegalArgumentException.class, () -> {
			LongArg.create();
		});

		assertThrows(IllegalArgumentException.class, () -> {
			LongArg.create(new long[0]);
		});

		// Check create output-array with invalid size
		assertThrows(IllegalArgumentException.class, () -> {
			LongArg.createOutput(0);
		});

		assertThrows(IllegalArgumentException.class, () -> {
			LongArg.createOutput(-1);
		});

		assertThrows(IllegalArgumentException.class, () -> {
			LongArg.createOutput(Integer.MIN_VALUE);
		});
	}

	@Test
	void testLongSingle() {
		KernelArg inArg;

		// Create KernelArgs
		inArg = LongArg.createValue((long) 4);
		outArg = LongArg.createOutput(1);

		cpLongSingle.launch(inArg, outArg);

		// Check result
		assertEquals(1, outArg.asLongArray().length);
		assertEquals(4, outArg.asLongArray()[0]);

		// Create KernelArgs
		inArg = LongArg.createValue((long) -128);

		cpLongSingle.launch(inArg, outArg);

		// Check result
		assertEquals(1, outArg.asLongArray().length);
		assertEquals((long) -128, outArg.asLongArray()[0]);
	}

	@Test
	void testLongArray() {
		// Check test-Arrays should be correct
		checkTestArray0(testArray0);
		checkTestArray1(testArray1);

		// Create KernelArgs (download both)
		inArg = LongArg.create(testArray0, true);
		outArg = LongArg.create(testArray1, true);

		cpLongArray.launch(inArg, outArg);

		// Check Result
		checkTestArray0(inArg.asLongArray());
		checkTestArray0(outArg.asLongArray());
		// Other Array should be unchanged
		checkTestArray0(testArray0);
		checkTestArray1(testArray1);

		// Create KernelArgs (download only inArg)
		inArg = LongArg.create(testArray0, true);
		outArg = LongArg.create(testArray1, false);

		cpLongArray.launch(inArg, outArg);

		// Check Result
		checkTestArray0(inArg.asLongArray());
		checkTestArray1(outArg.asLongArray());
		// Other Array should be unchanged
		checkTestArray0(testArray0);
		checkTestArray1(testArray1);

		// Create KernelArgs (download only outArg)
		inArg = LongArg.create(testArray0, false);
		outArg = LongArg.create(testArray1, true);

		cpLongArray.launch(inArg, outArg);

		// Check Result
		checkTestArray0(inArg.asLongArray());
		checkTestArray0(outArg.asLongArray());
		// Other Array should be unchanged
		checkTestArray0(testArray0);
		checkTestArray1(testArray1);

		// Create KernelArgs (download nothing)
		inArg = LongArg.create(testArray0, false);
		outArg = LongArg.create(testArray1, false);

		cpLongArray.launch(inArg, outArg);

		// Check Result
		checkTestArray0(inArg.asLongArray());
		checkTestArray1(outArg.asLongArray());
		// Other Array should be unchanged
		checkTestArray0(testArray0);
		checkTestArray1(testArray1);
	}

	@Test
	void testLongOutput() {
		// Check test-Arrays should be correct
		checkTestArray0(testArray0);
		checkTestArray1(testArray1);

		// Create KernelArgs
		inArg = LongArg.create(testArray0);
		outArg = LongArg.createOutput(n);

		cpLongArray.launch(inArg, outArg);

		// Check Result
		checkTestArray0(inArg.asLongArray());
		checkTestArray0(outArg.asLongArray());
		// Other Array should be unchanged
		checkTestArray0(testArray0);
		checkTestArray1(testArray1);

		// Create KernelArgs
		inArg = LongArg.create(testArray1);
		outArg = LongArg.createOutput(n);

		cpLongArray.launch(inArg, outArg);

		// Check Result
		checkTestArray1(inArg.asLongArray());
		checkTestArray1(outArg.asLongArray());
		// Other Array should be unchanged
		checkTestArray0(testArray0);
		checkTestArray1(testArray1);

		// Use the same KernelArgs again
		cpLongArray.launch(inArg, outArg);

		// Check Result
		checkTestArray1(inArg.asLongArray());
		checkTestArray1(outArg.asLongArray());
		// Other Array should be unchanged
		checkTestArray0(testArray0);
		checkTestArray1(testArray1);
	}
}
