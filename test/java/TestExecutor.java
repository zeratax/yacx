import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.jupiter.api.Assertions.assertThrows;

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
	
	static float a;
	static float[] x, y;
	static float[] result;
	static int n;
	static FloatArg aArg, xArg, yArg, outArg;
	static IntArg nArg;
	static Executor.KernelArgCreator creator;
	

	@BeforeAll
	static void init() throws IOException {
		//Init test-data
		a = 5.1f;
		n = 16*8;
		x = new float[n];
		y = new float[n];
		result = new float[n];
		
		for (int i = 0; i < n; i++) {
			x[i] = i-5;
			y[i] = (n - i);
			result[i] = a*x[i]+y[i];
		}
		
		grid = n/2;
		block = 2;
		grid0 = n/8;
		grid1 = 1;
		grid2 = 1;
		block0 = 8;
		block1 = 1;
		block2 = 1;
		options = Options.createOptions();
		device = Device.createDevice();
		devicename = device.getName();
		
		//init kernel-arguments
		aArg = FloatArg.create(a);
		xArg = FloatArg.create(x);
		yArg = FloatArg.create(y);
		nArg = IntArg.create(n);
		
		//KernelArg-Creator for benchmark-test
		creator = new Executor.KernelArgCreator() {
			@Override
			public int getDataLength(int dataSizeBytes) {
				return dataSizeBytes/FloatArg.SIZE_BYTES;
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
				//test data
				a = 5.2f;
				x = new float[dataLength];
				y = new float[dataLength];
				
				for (int i = 0; i < dataLength; i++) {
					x[i] = i-5;
					y[i] = (dataLength - i);
				}
				
				return new KernelArg[] {FloatArg.create(a), FloatArg.create(x), FloatArg.create(y),
											FloatArg.createOutput(dataLength), IntArg.create(n)};
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
	void testLaunch() throws IOException{
		//Initialize outArg before ervery run with 0
		outArg = FloatArg.create(new float[n], true);
		
		//Run every launch-Method with correct Arguments
		Executor.launch("saxpy", grid, block, aArg, xArg, yArg, outArg, nArg);
		
		//Check Result
		checkResult();
		
		//Other Methods...
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
		
		//Methods with more grid and block parameters
		outArg = FloatArg.create(new float[n], true);
		Executor.launch("saxpy", options, grid0, grid1, grid2, block0, block1, block2, aArg, xArg, yArg, outArg, nArg);
		checkResult();
		
		outArg = FloatArg.create(new float[n], true);
		Executor.launch("saxpy", options, devicename, grid0, grid1, grid2, block0, block1, block2, aArg, xArg, yArg, outArg, nArg);
		checkResult();
		
		outArg = FloatArg.create(new float[n], true);
		Executor.launch(saxpy, "saxpy", grid0, grid1, grid2, block0, block1, block2, aArg, xArg, yArg, outArg, nArg);
		checkResult();
		
		outArg = FloatArg.create(new float[n], true);
		Executor.launch(saxpy, "saxpy", options, grid0, grid1, grid2, block0, block1, block2, aArg, xArg, yArg, outArg, nArg);
		checkResult();
		
		outArg = FloatArg.create(new float[n], true);
		Executor.launch(saxpy, "saxpy", options, devicename, grid0, grid1, grid2, block0, block1, block2, aArg, xArg, yArg, outArg, nArg);
		checkResult();
	}
	
	@Test
	void testBenchmarkInvalid() {
		//Invalid number of executions
		assertThrows(IllegalArgumentException.class, () -> {
			Executor.benchmark(saxpy, "saxpy", options, 0, creator, 1024, 2048);
		});
		
		//Invalid dataSize
		assertThrows(IllegalArgumentException.class, () -> {
			Executor.benchmark(saxpy, "saxpy", options, 3, creator, 1024, -1, 2048);
		});
		
		//null
		assertThrows(NullPointerException.class, () -> {
			Executor.benchmark(saxpy, null, options, 3, creator, 1024, 2048);
		});
		
		//invalid KernelArgCreator
		assertThrows(IllegalArgumentException.class, () -> {
			Executor.benchmark(saxpy, "saxpy", options, 3, new Executor.KernelArgCreator() {
				
				@Override
				public int getDataLength(int dataSizeBytes) {
					return creator.getDataLength(dataSizeBytes);
				}
				
				@Override
				public int getGrid0(int dataLength) {
					return creator.getGrid0(dataLength);
				}
				
				@Override
				public int getBlock0(int dataLength) {
					return 0; //Invalid Number of Blocks
				}
				
				@Override
				public KernelArg[] createArgs(int dataLength) {
					return creator.createArgs(dataLength);
				}
			}, 1024, 2048);
		});
		
		//Invalid KernelArgCreator if dataSize > 2048 Bytes
		Executor.KernelArgCreator creatorInvalid = new Executor.KernelArgCreator() {
			
			@Override
			public int getDataLength(int dataSizeBytes) {
				return creator.getDataLength(dataSizeBytes);
			}
			
			@Override
			public int getGrid0(int dataLength) {
				return creator.getGrid0(dataLength);
			}
			
			@Override
			public int getBlock0(int dataLength) {
				return creator.getBlock0(dataLength);
			}
			
			@Override
			public KernelArg[] createArgs(int dataLength) {
				if (dataLength <= 512) {
					return creator.createArgs(dataLength);
				} else {
					//Invalid KernelArgs
					return new KernelArg[] {aArg, null, yArg, nArg};
				}
			}
		};
		
		Executor.BenchmarkResult result = Executor.benchmark(saxpy, "saxpy", options, 3, creatorInvalid, 1024, 2048);
		assertNotNull(result);
		
		//Run with more than 2048 dataSize
		assertThrows(NullPointerException.class, () -> {
			Executor.benchmark(saxpy, "saxpy", options, 3, creatorInvalid, 2048, 4096);
		});
	}
	
	@Test
	void testBenchmarkValid() {
		//Run benchmark-tests correctly
		Executor.BenchmarkResult result = Executor.benchmark(saxpy, "saxpy", options, 3, creator, 1024, 2048);
		assertNotNull(result);
		
		result = Executor.benchmark(saxpy, "saxpy", options, 7, creator, 512, 768);
		assertNotNull(result);
	}
}
