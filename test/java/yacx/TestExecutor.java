package yacx;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

class TestExecutor extends TestJNI {
	final static float DELTA = (float) 10e-5;

	static int grid, block;
	static int grid0, grid1, grid2, block0, block1, block2;
	static Options options;
	static Device device;
	static String devicename;
	static String saxpy, filterk;

	static float a;
	static float[] x, y;
	static float[] result;
	static int n;
	static FloatArg xArg, yArg, outArg;
	static KernelArg aArg, nArg;
	static Executor.KernelArgCreator creatorSaxpy, creatorFilter;

	@BeforeAll
	static void init() throws IOException {
		// Load Saxpy and Filter-Kernel as String
		saxpy = Utils.loadFile("kernels/saxpy.cu");
		filterk = Utils.loadFile("kernels/filter_k.cu");

		// Init test-data
		a = 5.1f;
		n = 16 * 8;
		x = new float[n];
		y = new float[n];
		result = new float[n];

		for (int i = 0; i < n; i++) {
			x[i] = i - 5;
			y[i] = (n - i);
			result[i] = a * x[i] + y[i];
		}

		grid = n / 2;
		block = 2;
		grid0 = n / 8;
		grid1 = 1;
		grid2 = 1;
		block0 = 8;
		block1 = 1;
		block2 = 1;
		options = Options.createOptions();
		device = Devices.findDevice();
		devicename = device.getName();

		// init kernel-arguments
		aArg = FloatArg.createValue(a);
		xArg = FloatArg.create(x);
		yArg = FloatArg.create(y);
		nArg = IntArg.createValue(n);

		// KernelArg-Creator for saxpy-benchmark-test
		creatorSaxpy = new Executor.KernelArgCreator() {
			@Override
			public int getDataLength(int dataSizeBytes) {
				return (int) (dataSizeBytes / FloatArg.SIZE_BYTES);
			}

			@Override
			public int getGrid0(int dataLength) {
				return dataLength;
			}

			@Override
			public int getBlock0(int dataLength) {
				return 1;
			}

			@Override
			public KernelArg[] createArgs(int dataLength) {
				// test data
				a = 5.2f;
				x = new float[dataLength];
				y = new float[dataLength];

				for (int i = 0; i < dataLength; i++) {
					x[i] = i - 5;
					y[i] = (dataLength - i);
				}

				return new KernelArg[] { FloatArg.createValue(a), FloatArg.create(x), FloatArg.create(y),
						FloatArg.createOutput(dataLength), IntArg.createValue(n) };
			}
		};

		// KernelArg-Creator for filter-benchmark-test
		creatorFilter = new Executor.KernelArgCreator() {

			@Override
			public int getDataLength(int dataSizeBytes) {
				return (int) (dataSizeBytes / IntArg.SIZE_BYTES);
			}

			@Override
			public int getGrid0(int dataLength) {
				return dataLength;
			}

			@Override
			public int getBlock0(int dataLength) {
				return 1;
			}

			@Override
			public KernelArg[] createArgs(int dataLength) {
				int[] in = new int[dataLength];

				for (int i = 0; i < dataLength; i++) {
					in[i] = i;
				}

				return new KernelArg[] { IntArg.createOutput(dataLength / 2), IntArg.create(new int[] { 0 }, true),
						IntArg.create(in), IntArg.create(dataLength) };
			}
		};
	}

	/**
	 * Check result after launching saxpy kernel
	 */
	void checkResult() {
		float[] out = outArg.asFloatArray();

		assertTrue(out.length == n);

		for (int i = 0; i < n; i++) {
			assertTrue(Math.abs(out[i] - result[i]) < DELTA);
		}
	}

	@Test
	void testLaunch() throws IOException {
		// Initialize outArg before ervery run with 0
		outArg = FloatArg.create(new float[n], true);

		// Run every launch-Method with correct Arguments
		Executor.launch("saxpy", grid, block, aArg, xArg, yArg, outArg, nArg);

		// Check Result
		checkResult();

		// Other Methods...
		outArg = FloatArg.create(new float[n], true);
		Executor.launch("saxpy", options, grid, block, aArg, xArg, yArg, outArg, nArg);
		checkResult();

		outArg = FloatArg.create(new float[n], true);
		Executor.launch("saxpy", options, devicename, grid, block, aArg, xArg, yArg, outArg, nArg);
		checkResult();

		outArg = FloatArg.create(new float[n], true);
		Executor.launch(saxpy, "saxpy", grid, block, aArg, xArg, yArg, outArg, nArg);
		checkResult();

		outArg = FloatArg.create(new float[n], true);
		Executor.launch(saxpy, "saxpy", options, grid, block, aArg, xArg, yArg, outArg, nArg);
		checkResult();

		outArg = FloatArg.create(new float[n], true);
		Executor.launch(saxpy, "saxpy", options, devicename, grid, block, aArg, xArg, yArg, outArg, nArg);
		checkResult();

		// Methods with more grid and block parameters
		outArg = FloatArg.create(new float[n], true);
		Executor.launch(saxpy, "saxpy", grid0, grid1, grid2, block0, block1, block2, aArg, xArg, yArg, outArg, nArg);
		checkResult();

		outArg = FloatArg.create(new float[n], true);
		Executor.launch(saxpy, "saxpy", options, grid0, grid1, grid2, block0, block1, block2, aArg, xArg, yArg, outArg,
				nArg);
		checkResult();

		outArg = FloatArg.create(new float[n], true);
		Executor.launch(saxpy, "saxpy", options, devicename, grid0, grid1, grid2, block0, block1, block2, aArg, xArg,
				yArg, outArg, nArg);
		checkResult();
	}

	@Test
	void testLaunchTemplate() throws IOException {
		// Kernel with one template argument
		String kernelString = Utils.loadFile("kernels/template.cu");
		String kernelName = "f3";

		// Argument for result
		IntArg resultArg = IntArg.createOutput(1);

		// Try to run kernel without passing template parameter
		assertThrows(IllegalArgumentException.class, () -> {
			Executor.launch(kernelString, kernelName, Options.createOptions(), Devices.findDevice().getName(),
					new String[0], 1, 1, resultArg);
		});

		String templateParameter;

		// Run with different template parameters
		templateParameter = "int";

		Executor.launch(kernelString, kernelName, Options.createOptions(), Devices.findDevice().getName(),
				new String[] { templateParameter }, 1, 1, resultArg);

		assertEquals(IntArg.SIZE_BYTES, resultArg.asIntArray()[0],
				"an int in CUDA is not " + IntArg.SIZE_BYTES + " bytes long");

		templateParameter = "long";

		Executor.launch(kernelString, kernelName, Options.createOptions(), Devices.findDevice().getName(),
				new String[] { templateParameter }, 1, 1, resultArg);

		assertEquals(LongArg.SIZE_BYTES, resultArg.asIntArray()[0],
				"an long in CUDA is not " + LongArg.SIZE_BYTES + " bytes long");

		// Use the other template-launch-method
		templateParameter = "double";

		Executor.launch(kernelString, kernelName, Options.createOptions(), Devices.findDevice().getName(),
				new String[] { templateParameter }, 1, 1, 1, 1, 1, 1, resultArg);

		assertEquals(DoubleArg.SIZE_BYTES, resultArg.asIntArray()[0],
				"an double in CUDA is not " + DoubleArg.SIZE_BYTES + " bytes long");
	}

	@Test
	void testBenchmarkInvalid() {
		// Invalid number of executions
		assertThrows(IllegalArgumentException.class, () -> {
			Executor.benchmark(saxpy, "saxpy", options, Devices.findDevice(), 0, creatorSaxpy, 1024, 2048);
		});

		// Invalid dataSize
		assertThrows(IllegalArgumentException.class, () -> {
			Executor.benchmark(saxpy, "saxpy", options, Devices.findDevice(), 3, creatorSaxpy, 1024, -1, 2048);
		});

		// null
		assertThrows(NullPointerException.class, () -> {
			Executor.benchmark(saxpy, null, options, Devices.findDevice(), 3, creatorSaxpy, 1024, 2048);
		});

		// invalid KernelArgCreator
		assertThrows(IllegalArgumentException.class, () -> {
			Executor.benchmark(saxpy, "saxpy", options, Devices.findDevice(), 3, new Executor.KernelArgCreator() {

				@Override
				public int getDataLength(int dataSizeBytes) {
					return creatorSaxpy.getDataLength(dataSizeBytes);
				}

				@Override
				public int getGrid0(int dataLength) {
					return creatorSaxpy.getGrid0(dataLength);
				}

				@Override
				public int getBlock0(int dataLength) {
					return 0; // Invalid Number of Blocks
				}

				@Override
				public KernelArg[] createArgs(int dataLength) {
					return creatorSaxpy.createArgs(dataLength);
				}
			}, 1024, 2048);
		});

		// Invalid KernelArgCreator if dataSize > 2048 Bytes
		Executor.KernelArgCreator creatorInvalid = new Executor.KernelArgCreator() {

			@Override
			public int getDataLength(int dataSizeBytes) {
				return creatorSaxpy.getDataLength(dataSizeBytes);
			}

			@Override
			public int getGrid0(int dataLength) {
				return creatorSaxpy.getGrid0(dataLength);
			}

			@Override
			public int getBlock0(int dataLength) {
				return creatorSaxpy.getBlock0(dataLength);
			}

			@Override
			public KernelArg[] createArgs(int dataLength) {
				if (dataLength <= 512) {
					return creatorSaxpy.createArgs(dataLength);
				} else {
					// Invalid KernelArgs
					return new KernelArg[] { aArg, null, yArg, nArg };
				}
			}
		};

		Executor.BenchmarkResult result = Executor.benchmark(saxpy, "saxpy", options, Devices.findDevice(), 3,
				creatorInvalid, 1024, 2048);
		assertNotNull(result);

		// Run with more than 2048 dataSize
		assertThrows(NullPointerException.class, () -> {
			Executor.benchmark(saxpy, "saxpy", options, Devices.findDevice(), 3, creatorInvalid, 2048, 4096);
		});
	}

	@Test
	void testBenchmarkValid() throws IOException {
		// Run saxpy-benchmark-tests correctly
		Executor.BenchmarkResult result = Executor.benchmark("saxpy", options, 3, creatorSaxpy, 1024, 2048);
		assertNotNull(result);

		// Other saxpy-benchmark-test
		result = Executor.benchmark(saxpy, "saxpy", options, Devices.findDevice(), 7, creatorSaxpy, 512, 768);
		assertNotNull(result);

		// Run filter-benchmark-tests correctly
		result = Executor.benchmark(filterk, "filter_k", options, Devices.findDevice(), 4, creatorFilter, 1024, 2048,
				4096);
		assertNotNull(result);
	}
}
