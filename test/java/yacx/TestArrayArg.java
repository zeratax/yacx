package yacx;

import static org.junit.jupiter.api.Assertions.*;

import java.io.IOException;
import java.util.Arrays;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

public class TestArrayArg extends TestJNI {
	static String saxpy, filterk;
	ArrayArg arg;

	@BeforeAll
	static void loadSaxpyFilter() throws IOException {
		// Load Saxpy and Filter-Kernel as String
		saxpy = Utils.loadFile("kernels/saxpy.cu");
		filterk = Utils.loadFile("kernels/filter_k.cu");
	}

	@Test
	void testGetLength() {
		arg = BooleanArg.createOutput(16);
		assertEquals(16, arg.getLength());

		arg = ByteArg.create((byte) 16);
		assertEquals(1, arg.getLength());

		arg = ShortArg.createOutput(27);
		assertEquals(27, arg.getLength());

		arg = IntArg.create(17, 18, 20);
		assertEquals(3, arg.getLength());

		arg = LongArg.create(17, 18, 20);
		assertEquals(3, arg.getLength());

		arg = FloatArg.create(17, 18);
		assertEquals(2, arg.getLength());

		arg = DoubleArg.create(17, 18, 20);
		assertEquals(3, arg.getLength());
	}

	@Test
	void testGetSetDownloadUpload() {
		arg = BooleanArg.create(true);
		assertTrue(arg.isUpload());
		assertFalse(arg.isDownload());

		arg.setUpload(true);
		assertTrue(arg.isUpload());
		assertFalse(arg.isDownload());

		arg.setUpload(false);
		assertFalse(arg.isUpload());
		assertFalse(arg.isDownload());

		arg.setDownload(true);
		assertFalse(arg.isUpload());
		assertTrue(arg.isDownload());

		arg = IntArg.create(new int[] { 18 }, false);
		assertTrue(arg.isUpload());
		assertFalse(arg.isDownload());

		arg.setDownload(true);
		assertTrue(arg.isUpload());
		assertTrue(arg.isDownload());

		arg = IntArg.create(new int[] { 18 }, true);
		assertTrue(arg.isUpload());
		assertTrue(arg.isDownload());

		arg = IntArg.createOutput(3);
		assertTrue(arg.isDownload());
		assertFalse(arg.isUpload());

		// Test run filter-kernel with arg as outputarg
		IntArg counterArg = IntArg.create(0);
		Executor.launch(filterk, "filter_k", 6, 1, arg, counterArg, IntArg.create(1, 2, 3, 4, 5, 6),
				IntArg.createValue(6));
		assertTrue(arg.isDownload());
		assertFalse(arg.isUpload());

		// and make a second run with arg as inputarg
		arg.setUpload(true);
		arg.setDownload(false);
		assertFalse(arg.isDownload());
		assertTrue(arg.isUpload());

		IntArg outputArg = IntArg.createOutput(3);
		Executor.launch(filterk, "filter_k", 3, 1, outputArg, counterArg, arg, IntArg.createValue(3));

		assertEquals(3, outputArg.getLength());
		assertEquals(3, outputArg.asIntArray().length);
		for (int i = 0; i < 3; i++) {
			assertTrue(Arrays.binarySearch(new int[] { 1, 3, 5 }, outputArg.asIntArray()[i]) >= 0);
		}

		assertFalse(arg.isDownload());
		assertTrue(arg.isUpload());
	}

	@Test
	void testSlice() {
		boolean[] testArray = new boolean[20];
		for (int i = 0; i < 20; i++) {
			if (i < 10)
				testArray[i] = true;
			else
				testArray[i] = false;
		}

		BooleanArg arg = BooleanArg.create(testArray);
		assertEquals(testArray.length, arg.getLength());
		assertArrayEquals(testArray, arg.asBooleanArray());

		// Test invalid slices
		assertThrows(IllegalArgumentException.class, () -> {
			arg.slice(-1, 10);
		});

		assertThrows(IllegalArgumentException.class, () -> {
			arg.slice(0, 25);
		});

		assertThrows(IllegalArgumentException.class, () -> {
			arg.slice(5, 21);
		});

		assertThrows(IllegalArgumentException.class, () -> {
			arg.slice(7, 3);
		});

		assertThrows(IllegalArgumentException.class, () -> {
			arg.slice(7, 6);
		});

		// Slice elements 7,8,9,10,11,12
		BooleanArg arg2 = arg.slice(7, 12);
		// Should be unchanged
		assertEquals(testArray.length, arg.getLength());
		assertArrayEquals(testArray, arg.asBooleanArray());

		// The slice should be survivable without original arg
		arg.dispose();

		assertEquals(6, arg2.getLength());
		assertArrayEquals(new boolean[] { true, true, true, false, false, false }, arg2.asBooleanArray());

		// Slice sliced arg elemnts 2,3 (so elements 9,10 from originally arg)
		BooleanArg arg3 = arg2.slice(2, 3);
		// Should be unchanged
		assertEquals(6, arg2.getLength());
		assertArrayEquals(new boolean[] { true, true, true, false, false, false }, arg2.asBooleanArray());

		assertEquals(2, arg3.getLength());
		assertArrayEquals(new boolean[] { true, false }, arg3.asBooleanArray());
	}

	@Test
	void testSlice2() {
		// Kernel for copy override intarray with ones
		String writeTwosString = "extern \"C\" __global__\n" + "void writeTwos(int* array) {\n"
				+ "  int i = (blockIdx.x * blockDim.x) + threadIdx.x;\n" + "  array[i] = 2;\n" + "}\n" + "";

		// Run filter kernel and use sliced output as in- and output for writeTwos

		IntArg srcArg = IntArg.create(1, 2, 3, 4, 5, 6);
		IntArg outputArg = IntArg.createOutput(srcArg.getLength());
		IntArg counterArg = IntArg.create(new int[] { 0 }, true);
		KernelArg nArg = IntArg.createValue(srcArg.getLength());
		Executor.launch(filterk, "filter_k", srcArg.getLength(), 1, outputArg, counterArg, srcArg, nArg);

		// Check result
		assertEquals(6, outputArg.getLength());
		assertEquals(3, counterArg.asIntArray()[0]);
		for (int i = 0; i < 3; i++) {
			assertTrue(Arrays.binarySearch(new int[] { 1, 3, 5 }, outputArg.asIntArray()[i]) >= 0);
		}

		// Slice output
		srcArg = outputArg.slice(1, 2);
		srcArg.setUpload(true);
		// argument should be uploaded and downloaded
		assertTrue(srcArg.isUpload());
		assertTrue(srcArg.isDownload());

		// Check sliced arg
		assertEquals(2, srcArg.getLength());
		for (int i = 0; i < 2; i++) {
			assertEquals(outputArg.asIntArray()[i + 1], srcArg.asIntArray()[i]);
		}

		// Run writeTwos
		Executor.launch(writeTwosString, "writeTwos", 2, 1, srcArg);

		// Check result
		assertEquals(2, srcArg.getLength());
		assertArrayEquals(new int[] { 2, 2 }, srcArg.asIntArray());

		// Originally should be changed too
		assertEquals(6, outputArg.getLength());
		assertTrue(Arrays.binarySearch(new int[] { 1, 3, 5 }, outputArg.asIntArray()[0]) >= 0);
		assertEquals(2, outputArg.asIntArray()[1]);
		assertEquals(2, outputArg.asIntArray()[2]);
	}
}
