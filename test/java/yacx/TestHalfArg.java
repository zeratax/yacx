package yacx;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

public class TestHalfArg extends TestJNI {
	static final String checkHalfs = "#include <cuda_fp16.h>\n" + "extern \"C\"__global__\n"
			+ "void checkFtoHSeq(float* floats, __half* halfs, int* counter, int n){\n" + "    int i = threadIdx.x;\n"
			+ "    if (__heq(__float2half(floats[i]), halfs[i])){\n" + "        atomicAdd(counter, 1);\n" + "    }\n"
			+ "}";
	static final String checkFloats = "#include <cuda_fp16.h>\n" + "extern \"C\" __global__\n"
			+ "void checkHtoFSeq(float* floats, __half* halfs, int* counter, int n){\n" + "    int i = threadIdx.x;\n"
			+ "    if (__half2float(halfs[i]) == floats[i]){\n" + "        atomicAdd(counter, 1);\n" + "    }\n" + "}";

	static Kernel cpHalfArray, cpHalfSingle;
	static int n, dimN0, dimM0, dimN1, dimM1;
	// test-data as floats
	static float[] testArray0, testArray1, testArrayMatrix0, testArrayMatrix1;
	// test-data as floats after conversion to half
	static float[] testArray0Half, testArray1Half, testMatrix0Half, testMatrix1Half;
	HalfArg inArg, outArg;

	@BeforeAll
	static void init() {
		// CUDA compute-version 6 required
		org.junit.Assume.assumeTrue("CUDA compute-version 6 or higher required",
				Devices.findDevice().getMajorVersion() >= 6);

		// test-data as float-arrays
		n = 19;
		dimN0 = 19;
		dimM0 = 19;
		dimN1 = 18;
		dimM1 = 37;
		testArray0 = new float[n];
		testArray1 = new float[n];
		testArrayMatrix0 = new float[dimN0 * dimM0];
		testArrayMatrix0 = new float[dimN1 * dimM1];

		for (int i = 0; i < n; i++) {
			testArray0[i] = 1.225475432416689f * i;
			testArray1[i] = 13.4f + (n - i) * 0.89999f;
		}

		for (int i = 0; i < dimN0 * dimM0; i++) {
			testArrayMatrix0[i] = 17.21838232422754f * i;
		}
		for (int i = 0; i < dimN1 * dimM1; i++) {
			testArrayMatrix1[i] = 143.4f + i * -23.9999f;
		}

		// convert test-data to half-arrays
		HalfArg testArray0Half = HalfArg.create(testArray0);
		HalfArg testArray1Half = HalfArg.create(testArray1);
		TestHalfArg.testArray0Half = testArray0Half.asFloatArray();
		TestHalfArg.testArray1Half = testArray1Half.asFloatArray();
		// check result using CUDA conversion from floats to halfs to floats
		checkHalfs(testArray0, testArray0Half);
		checkHalfs(testArray1, testArray1Half);
		checkFloats(testArray0, TestHalfArg.testArray0Half);
		checkFloats(testArray1, TestHalfArg.testArray1Half);

		// Kernel for copy half* in to half* out
		String copyHalfArrayString = "#include <cuda_fp16.h>\n" + "extern \"C\" __global__\n"
				+ "void copyHalf(__half* in, __half* out) {\n" + "  int i = (blockIdx.x * blockDim.x) + threadIdx.x;\n"
				+ "  out[i] = in[i];\n" + "}\n" + "";

		cpHalfArray = Program.create(copyHalfArrayString, "copyHalf").compile();
		// Configure with kernel n Threads
		cpHalfArray.configure(n, 1);

		// Kernel for copy a float-value
		String copyHalfString = "#include <cuda_fp16.h>\n" + "extern \"C\" __global__\n"
				+ "void copyHalf(__half in, __half* out) {\n" + "  *out = in;\n" + "}\n" + "";

		cpHalfSingle = Program.create(copyHalfString, "copyHalf").compile();
		// Configure Kernel with 1 thread
		cpHalfSingle.configure(1, 1);
	}

	/**
	 * Check if testArray is expected testArray0 after conversion to half and back
	 * to float.
	 * 
	 * @param testArray the array to be tested
	 */
	void checkTestArray0H(float[] testArray) {
		assertEquals(n, testArray.length);

		for (int i = 0; i < n; i++) {
			assertEquals(testArray0Half[i], testArray[i]);
		}
	}

	/**
	 * Check if testArray is expected testArray0.
	 * 
	 * @param testArray the array to be tested
	 */
	void checkTestArray0F(float[] testArray) {
		assertEquals(n, testArray.length);

		for (int i = 0; i < n; i++) {
			assertEquals(1.225475432416689f * i, testArray[i]);
		}
	}

	/**
	 * Check if testArray is expected testArray1 after conversion to half and back
	 * to float.
	 * 
	 * @param testArray the array to be tested
	 */
	void checkTestArray1H(float[] testArray) {
		assertEquals(n, testArray.length);

		for (int i = 0; i < n; i++) {
			assertEquals(testArray1Half[i], testArray[i]);
		}
	}

	/**
	 * Check if testArray is expected testArray1.
	 * 
	 * @param testArray the array to be tested
	 */
	void checkTestArray1F(float[] testArray) {
		assertEquals(n, testArray.length);

		for (int i = 0; i < n; i++) {
			assertEquals(13.4f + (n - i) * 0.89999f, testArray[i]);
		}
	}

	/**
	 * Checks if the passed float-array is correctly converted to a half-array and
	 * back.
	 */
	static void checkFloats(float[] floats, float[] convertedFloats) {
		assertEquals(floats.length, convertedFloats.length);

		// convert floats to halfs
		HalfArg halfs = HalfArg.create(floats);
		// check if conversion was correctly
		checkHalfs(floats, halfs);

		KernelArg nArg = IntArg.createValue(floats.length);
		// Counter for false converted floats
		IntArg counterArg = IntArg.create(new int[] { 0 }, true);

		Executor.launch(checkFloats, "checkHtoFSeq", Options.createOptions("--gpu-architecture=compute_60"), 1,
				floats.length, FloatArg.create(convertedFloats), halfs, counterArg, nArg);
		assertEquals(1, counterArg.getLength());
		assertEquals(floats.length, counterArg.asIntArray()[0]);
	}

	/**
	 * Checks if the passed float-array is correctly converted to a half-array.
	 */
	static void checkHalfs(float[] floats, HalfArg halfs) {
		KernelArg nArg = IntArg.createValue(floats.length);
		// Counter for false converted floats
		IntArg counterArg = IntArg.create(new int[] { 0 }, true);

		Executor.launch(checkHalfs, "checkFtoHSeq", Options.createOptions("--gpu-architecture=compute_60"), 1,
				floats.length, FloatArg.create(floats), halfs, counterArg, nArg);
		assertEquals(1, counterArg.getLength());
		assertEquals(floats.length, counterArg.asIntArray()[0]);
	}

	@Test
	void testInvalidParameter() {
		// Check if parameter is null
		assertThrows(NullPointerException.class, () -> {
			HalfArg.create((float[]) null);
		});

		// Check without any parameters
		assertThrows(IllegalArgumentException.class, () -> {
			HalfArg.create();
		});

		assertThrows(IllegalArgumentException.class, () -> {
			HalfArg.create(new float[0]);
		});

		// Check create output-array with invalid size
		assertThrows(IllegalArgumentException.class, () -> {
			HalfArg.createOutput(0);
		});

		assertThrows(IllegalArgumentException.class, () -> {
			HalfArg.createOutput(-1);
		});

		assertThrows(IllegalArgumentException.class, () -> {
			HalfArg.createOutput(Integer.MIN_VALUE);
		});
	}

	@Test
	void testHalfSingle() {
		KernelArg inArg;

		// Create KernelArgs
		inArg = HalfArg.createValue(4.9f);
		outArg = HalfArg.createOutput(1);

		cpHalfSingle.launch(inArg, outArg);

		// Check result
		assertEquals(1, outArg.asFloatArray().length);
		assertEquals(4.8984375f, outArg.asFloatArray()[0]);

		// Create KernelArgs
		inArg = HalfArg.createValue(-128.1f);

		cpHalfSingle.launch(inArg, outArg);

		// Check result
		assertEquals(1, outArg.asFloatArray().length);
		assertEquals(-128.125f, outArg.asFloatArray()[0]);
	}

	@Test
	void testHalfArray() {
		// Check test-Arrays should be correct
		checkTestArray0F(testArray0);
		checkTestArray1F(testArray1);

		// Create KernelArgs (download both)
		inArg = HalfArg.create(testArray0, true);
		outArg = HalfArg.create(testArray1, true);

		cpHalfArray.launch(inArg, outArg);

		// Check Result
		checkTestArray0H(inArg.asFloatArray());
		checkTestArray0H(outArg.asFloatArray());
		// Other Array should be unchanged
		checkTestArray0F(testArray0);
		checkTestArray1F(testArray1);

		// Create KernelArgs (download only inArg)
		inArg = HalfArg.create(testArray0, true);
		outArg = HalfArg.create(testArray1, false);

		cpHalfArray.launch(inArg, outArg);

		// Check Result
		checkTestArray0H(inArg.asFloatArray());
		checkTestArray1H(outArg.asFloatArray());
		// Other Array should be unchanged
		checkTestArray0F(testArray0);
		checkTestArray1F(testArray1);

		// Create KernelArgs (download only outArg)
		inArg = HalfArg.create(testArray0, false);
		outArg = HalfArg.create(testArray1, true);

		cpHalfArray.launch(inArg, outArg);

		// Check Result
		checkTestArray0H(inArg.asFloatArray());
		checkTestArray0H(outArg.asFloatArray());
		// Other Array should be unchanged
		checkTestArray0F(testArray0);
		checkTestArray1F(testArray1);

		// Create KernelArgs (download nothing)
		inArg = HalfArg.create(testArray0, false);
		outArg = HalfArg.create(testArray1, false);

		cpHalfArray.launch(inArg, outArg);

		// Check Result
		checkTestArray0H(inArg.asFloatArray());
		checkTestArray1H(outArg.asFloatArray());
		// Other Array should be unchanged
		checkTestArray0F(testArray0);
		checkTestArray1F(testArray1);
	}

	@Test
	void testHalfOutput() {
		// Check test-Arrays should be correct
		checkTestArray0F(testArray0);
		checkTestArray1F(testArray1);

		// Create KernelArgs
		inArg = HalfArg.create(testArray0);
		outArg = HalfArg.createOutput(n);

		cpHalfArray.launch(inArg, outArg);

		// Check Result
		checkTestArray0H(inArg.asFloatArray());
		checkTestArray0H(outArg.asFloatArray());
		// Other Array should be unchanged
		checkTestArray0F(testArray0);
		checkTestArray1F(testArray1);

		// Create KernelArgs
		inArg = HalfArg.create(testArray1);
		outArg = HalfArg.createOutput(n);

		cpHalfArray.launch(inArg, outArg);

		// Check Result
		checkTestArray1H(inArg.asFloatArray());
		checkTestArray1H(outArg.asFloatArray());
		// Other Array should be unchanged
		checkTestArray0F(testArray0);
		checkTestArray1F(testArray1);

		// Use the same KernelArgs again
		cpHalfArray.launch(inArg, outArg);

		// Check Result
		checkTestArray1H(inArg.asFloatArray());
		checkTestArray1H(outArg.asFloatArray());
		// Other Array should be unchanged
		checkTestArray0F(testArray0);
		checkTestArray1F(testArray1);
	}

	@Test
	void testToFloat() {
		FloatArg arg = HalfArg.create(testArray0).asFloatArg();

		checkFloats(testArray0, arg.asFloatArray());
		checkTestArray0F(testArray0);

		arg = HalfArg.create(testArray1).asFloatArg();

		checkFloats(testArray1, arg.asFloatArray());
		checkTestArray1F(testArray1);
	}

	@Test
	void testHalfArrayTransposed() {
		// Create KernelArgs
		inArg = HalfArg.createTransposed(testMatrix0Half, dimN0, dimM0);
		outArg = HalfArg.createOutput(dimN0 * dimM0);

		// Copy inArg to outArg
		cpHalfArray.launch(inArg, outArg);

		// Check result
		float[] testArray = outArg.asFloatArray();

		assertEquals(dimN0 * dimM0, testArray.length);

		for (int row = 0; row < dimN0; row++) {
			for (int column = 0; column < dimM0; column++) {
				assertEquals(testArray0Half[column * dimN0 + row], testArray[row * dimM0 + column]);
			}
		}

		// Same with testMatrix1
		inArg = HalfArg.createTransposed(testMatrix1Half, dimN1, dimM1);
		outArg = HalfArg.createOutput(dimN1 * dimM1);

		cpHalfArray.launch(inArg, outArg);

		testArray = outArg.asFloatArray();

		assertEquals(dimN1 * dimM1, testArray.length);

		for (int row = 0; row < dimN1; row++) {
			for (int column = 0; column < dimM1; column++) {
				assertEquals(testArray1Half[column * dimN1 + row], testArray[row * dimM1 + column]);
			}
		}
	}
}
