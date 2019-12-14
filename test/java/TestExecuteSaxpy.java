import static org.junit.Assert.assertTrue;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.io.IOException;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

public class TestExecuteSaxpy extends TestJNI {
	final float DELTA = (float) 10e-5;
	
	static float a, a2;
	static float[] x, x2, y, y2;
	static int n, n2;
	static FloatArg xArg, xArg2, yArg, yArg2, outArg, outArg2;
	static KernelArg aArg, aArg2, nArg, nArg2;
	
	@BeforeAll
	static void init() {
		//Init test-data
		a = 5.1f;
		a2 = -4.4f;
		n = 2*17;
		n2 = 19;
		x = new float[n];
		x2 = new float[n2];
		y = new float[n];
		y2 = new float[n2];
		
		for (int i = 0; i < n; i++) {
			x[i] = i-5;
			y[i] = (n - i);
		}
		
		for (int i = 0; i < n2; i++) {
			x2[i] = i-6;
			y2[i] = i;
		}
		
		//init kernel-arguments
		aArg = FloatArg.createValue(a);
		aArg2 = FloatArg.createValue(a2);
		xArg = FloatArg.create(x);
		xArg2 = FloatArg.create(x2);
		yArg = FloatArg.create(y);
		yArg2 = FloatArg.create(y2);
		outArg = FloatArg.createOutput(n);
		outArg2 = FloatArg.createOutput(n2);
		nArg = IntArg.createValue(n);
		nArg2 = IntArg.createValue(n2);
	}
	
	@Test
	void testSaxpyFilterInvalid() throws IOException {
		//Launch without arguments
		assertThrows(IllegalArgumentException.class, () -> {
			Executor.launch("saxpy", n, 1);
		});
		
		//TODO The errors are SIGSEVS :(
//		//Launch with to small number of arguments
//		assertThrows(ExecutorFailureException.class, () -> {
//			Executor.launch("saxpy", n, 1, aArg, xArg, yArg, outArg);
//		});
//		
//		//Launch with to much arguments
//		assertThrows(ExecutorFailureException.class, () -> {
//			Executor.launch("saxpy", n, 1, aArg, xArg, yArg, outArg, nArg, aArg);
//		});
//		
//		//Launch with arguments in false order
//		assertThrows(ExecutorFailureException.class, () -> {
//			Executor.launch("saxpy", n, 1, aArg, xArg, yArg, nArg, outArg);
//		});
	}
	
	@Test
	void testLaunchSaxpyValid() throws IOException{
		//Launch Kernel correctly
		Executor.launch("saxpy", n, 1, aArg, xArg, yArg, outArg, nArg);
		
		//Check Result
		float[] out = outArg.asFloatArray();
		assertEquals(n, out.length);
		for (int i = 0; i < n; i++) {
			assertTrue(Math.abs(a * x[i] + y[i] - out[i]) < DELTA);
		}
		
		//Launch Kernel correctly again
		Executor.launch("saxpy", n/2, 2, aArg, xArg, yArg, outArg, nArg);
		
		//Check Result
		out = outArg.asFloatArray();
		assertEquals(n, out.length);
		for (int i = 0; i < n; i++) {
			assertTrue(Math.abs(a * x[i] + y[i] - out[i]) < DELTA);
		}
		
		//Launch Kernel correctly again with different arguments
		Executor.launch("saxpy", n, 1, aArg2, xArg2, yArg2, outArg2, nArg2);
				
		//Check Result
		out = outArg2.asFloatArray();
		assertEquals(n2, out.length);
		for (int i = 0; i < n2; i++) {
			assertTrue(Math.abs(a2 * x2[i] + y2[i] - out[i]) < DELTA);
		}
	}
	
	@Test
	void testLaunchSaxpyValidLotsOfThreads() throws IOException{
		//Launch Kernel correctly again with lots of threads
		Executor.launch("saxpy", n*n, n, aArg, xArg, yArg, outArg, nArg);
		
		//Check Result
		float[] out = outArg.asFloatArray();
		assertEquals(n, out.length);
		for (int i = 0; i < n; i++) {
			assertTrue(Math.abs(a * x[i] + y[i] - out[i]) < DELTA);
		}
	}
	
	@Test
	void testLaunchSaxpyValidSmallNumberThreads() throws IOException{
		float[] array0 = new float[n];
		
		FloatArg outArg = FloatArg.create(array0, true);
		
		//Launch Kernel correctly again with to small number of threads
		Executor.launch("saxpy", n/2, 1, aArg, xArg, yArg, outArg, nArg);
		
		//Check Result
		float[] out = outArg.asFloatArray();
		assertEquals(n, out.length);
		for (int i = 0; i < n/2; i++) {
			assertTrue(Math.abs(a * x[i] + y[i] - out[i]) < DELTA);
		}
		//The other entrys should be unchanged 0
		for (int i = n/2; i < n; i++) {
			assertTrue(out[i] == 0);
		}
	}
}
