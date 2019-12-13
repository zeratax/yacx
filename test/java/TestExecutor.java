import static org.junit.Assert.assertTrue;

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
}
