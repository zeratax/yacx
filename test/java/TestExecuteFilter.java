import static org.junit.Assert.assertTrue;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.io.IOException;
import java.util.Arrays;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

public class TestExecuteFilter extends TestJNI {
	static float a, a2;
	static float[] x, x2, y, y2;
	static int n, n2;
	static IntArg srcArg, srcArg2, counterArg, outArg, outArg2, nArg, nArg2;
	
	@BeforeAll
	static void init() {
		//Init test-data
		n = 3*15;
		n2 = 17;
		int[] src = new int[n];
		int[] src2 = new int[n2];
		
		for (int i = 0; i < n; i++) {
			src[i] = i;
		}
		
		for (int i = 0; i < n2; i++) {
			src2[i] = 13 + (n-i)*3;
		}
		
		//init kernel-arguments
		srcArg = IntArg.create(src);
		srcArg2 = IntArg.create(src2);
		counterArg = IntArg.createOutput(1);
		outArg = IntArg.createOutput(n/2);
		outArg2 = IntArg.createOutput(n2/2);
		nArg = IntArg.create(n);
		nArg2 = IntArg.create(n2);
	}
	
	@Test
	void testLaunchFilterInvalid() throws IOException {
		//Launch without arguments
		assertThrows(IllegalArgumentException.class, () -> {
			Executor.launch("filter_k", n, 1);
		});
		
		//Launch with to small number of arguments
//		assertThrows(ExecutorFailureException.class, () -> {
//			Executor.launch("filter_k", n, 1, outArg, counterArg, srcArg);
//		});
//		
//		//Launch with to much arguments
//		assertThrows(ExecutorFailureException.class, () -> {
//			Executor.launch("filter_k", n, 1, outArg, counterArg, srcArg, nArg, outArg);
//		});
//		
//		//Launch with arguments in false order
//		assertThrows(ExecutorFailureException.class, () -> {
//			Executor.launch("filter_k", n, 1, counterArg, outArg, srcArg, nArg);
//		});
	}
	
	@Test
	void testLaunchFilterValid() throws IOException{
		//Launch Kernel correctly
		Executor.launch("filter_k", n, 1, outArg, counterArg, srcArg, nArg);
		
		//Check Result
		int[] out = outArg.asIntArray();
		assertTrue(out.length == n/2);
		for (int i = 0; i < n/2; i++) {
			assertTrue(out[i] % 2 == 1);
			assertTrue(Arrays.binarySearch(srcArg.asIntArray(), out[i]) >= 0);
		}
//
// 		//Launch Kernel correctly again
// 		Executor.launch("filter_k", n/3, 3, outArg, counterArg, srcArg, nArg);
//
// 		//Check Result
// 		out = outArg.asIntArray();
// 		assertTrue(out.length == n/2);
// 		for (int i = 0; i < n/2; i++) {
// 			assertTrue(out[i] % 2 == 1 && Arrays.binarySearch(srcArg.asIntArray(), out[i]) >= 0);
// 		}
//
// 		//Launch Kernel correctly again with different arguments
// 		Executor.launch("filter_k", n, 1, outArg2, counterArg, srcArg2, nArg2);
//
// 		//Check Result
// 		out = outArg2.asIntArray();
// 		assertTrue(out.length == n2/2);
// 		for (int i = 0; i < n2/2; i++) {
// 			assertTrue(out[i] % 2 == 1 && Arrays.binarySearch(srcArg2.asIntArray(), out[i]) >= 0);
// 		}
	}
	
// 	@Test
	void testLaunchFilterValidLotsOfThreads() throws IOException{
		//Launch Kernel correctly again with lots of threads
		Executor.launch("filter_k", n*n, n, outArg, counterArg, srcArg, nArg);
		
		//Check Result
		int[] out = outArg.asIntArray();
		assertTrue(out.length == n/2);
		for (int i = 0; i < n/2; i++) {
			assertTrue(out[i] % 2 == 1 && Arrays.binarySearch(srcArg.asIntArray(), out[i]) >= 0);
		}
	}
	
// 	@Test
	void testLaunchFilterValidSmallNumberThreads() throws IOException{
		int[] array0 = new int[n];
		
		outArg = IntArg.create(array0, true);
		
		//Launch Kernel correctly again with to small number of threads
		Executor.launch("filter_k", n/2, 1, outArg, counterArg, srcArg, nArg);
		
		//Check Result
		int[] out = outArg.asIntArray();
		for (int i = 0; i < n/4; i++) {
			assertTrue(out[i] % 2 == 1 && Arrays.binarySearch(srcArg.asIntArray(), out[i]) >= 0);
		}
		//The other entrys should be unchanged 0
		for (int i = n/4; i < n/2; i++) {
			assertTrue(out[i] == 0);
		}
	}
}