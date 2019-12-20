package yacx;

import static org.junit.Assert.assertTrue;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.io.IOException;
import java.util.Arrays;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

public class TestExecuteFilter extends TestJNI {
	static int[] src, src2;
	static int n, n2;
	static IntArg srcArg, srcArg2, outArg, outArg2;
	static KernelArg nArg, nArg2;
	
	@BeforeAll
	static void init() {
		//Init test-data
		n = 3*15;
		n2 = 17;
		src = new int[n];
		src2 = new int[n2];
		
		for (int i = 0; i < n; i++) {
			src[i] = i;
		}
		
		for (int i = 0; i < n2; i++) {
			src2[i] = 13 + (n-i)*3;
		}
		
		//init kernel-arguments
		srcArg = IntArg.create(src);
		srcArg2 = IntArg.create(src2);
		outArg = IntArg.createOutput(n/2);
		outArg2 = IntArg.createOutput(n2/2);
		nArg = IntArg.createValue(n);
		nArg2 = IntArg.createValue(n2);
		
		//Sort src-Arrays to use binarySearch later
		Arrays.sort(src);
		Arrays.sort(src2);
	}
	
	@Test
	void testLaunchFilterInvalid() throws IOException {
		//Launch without arguments
		assertThrows(IllegalArgumentException.class, () -> {
			Executor.launch("filter_k", n, 1);
		});
	}
	
	@Test
	void testLaunchFilterValid() throws IOException{
		//Initialize Counter-Arg
		IntArg counterArg = IntArg.create(new int[] {0}, true);
		
		//Launch Kernel correctly
		Executor.launch("filter_k", n, 1, outArg, counterArg, srcArg, nArg);
		
		//Check Result
		int[] out = outArg.asIntArray();
		assertEquals(n/2, counterArg.asIntArray()[0]);
		assertEquals(n/2, out.length);
		for (int i = 0; i < n/2; i++) {
			assertTrue(out[i] % 2 == 1);
			assertTrue(Arrays.binarySearch(src, out[i]) >= 0);
		}

		//Reset counterArg
		counterArg = IntArg.create(new int[] {0}, true);
		
		//Launch Kernel correctly again
 		Executor.launch("filter_k", n/3, 3, outArg, counterArg, srcArg, nArg);

 		//Check Result
 		out = outArg.asIntArray();
 		assertEquals(n/2, counterArg.asIntArray()[0]);
 		assertEquals(n/2, out.length);
 		for (int i = 0; i < n/2; i++) {
 			assertTrue(out[i] % 2 == 1);
 			assertTrue(Arrays.binarySearch(src, out[i]) >= 0);
 		}
 		
		//Reset counterArg
		counterArg = IntArg.create(new int[] {0}, true);

 		//Launch Kernel correctly again with different arguments
 		Executor.launch("filter_k", n2, 1, outArg2, counterArg, srcArg2, nArg2);

 		//Check Result
 		out = outArg2.asIntArray();
 		assertEquals(n2/2, counterArg.asIntArray()[0]);
 		assertEquals(n2/2, out.length);
 		for (int i = 0; i < n2/2; i++) {
 			assertTrue(out[i] % 2 == 1);
 			assertTrue(Arrays.binarySearch(src2, out[i]) >= 0);
 		}
 		
 		//Launch again without reset counter
 		//OutputArg must be bigger
 		IntArg outArg2 = IntArg.createOutput(n2);
 		
 		//Launch Kernel correctly again with different arguments
 		Executor.launch("filter_k", n2, 1, outArg2, counterArg, srcArg2, nArg2);

 		//Check Result
 		out = outArg2.asIntArray();
 		boolean odd = n2 % 2 == 1;
 		assertEquals(n2 - (odd ? 1 : 0), counterArg.asIntArray()[0]);
 		assertEquals(n2, out.length);
 		
 		//Check the last n/2 elements
 		for (int i = n/2; i < n2 - (odd ? 1 : 0); i++) {
 			assertTrue(out[i] % 2 == 1);
 			assertTrue(Arrays.binarySearch(src2, out[i]) >= 0);
 		}
	}
	
 	@Test
	void testLaunchFilterValidLotsOfThreads() throws IOException{
 		//Initialize Counter-Arg
 		IntArg counterArg = IntArg.create(new int[] {0}, true);
 				
		//Launch Kernel correctly again with lots of threads
		Executor.launch("filter_k", n*n, n, outArg, counterArg, srcArg, nArg);
		
		//Check Result
		int[] out = outArg.asIntArray();
		assertEquals(n/2, counterArg.asIntArray()[0]);
		assertEquals(n/2, out.length);
		for (int i = 0; i < n/2; i++) {
			assertTrue(out[i] % 2 == 1);
 			assertTrue(Arrays.binarySearch(src, out[i]) >= 0);
		}
	}
	
 	@Test
	void testLaunchFilterValidSmallNumberThreads() throws IOException{
 		//Initialize Counter-Arg
 		IntArg counterArg = IntArg.create(new int[] {0}, true);
 				
		int[] array0 = new int[n];
		
		IntArg outArg = IntArg.create(array0, true);
		
		//Launch Kernel correctly again with to small number of threads
		Executor.launch("filter_k", n/2, 1, outArg, counterArg, srcArg, nArg);
		
		//Check Result
		int[] out = outArg.asIntArray();
		assertEquals(n/4, counterArg.asIntArray()[0]);
		for (int i = 0; i < n/4; i++) {
			assertTrue(out[i] % 2 == 1);
 			assertTrue(Arrays.binarySearch(src, out[i]) >= 0);
		}
		//The other entrys should be unchanged 0
		for (int i = n/4; i < n/2; i++) {
			assertTrue(out[i] == 0);
		}
	}
}
