package yacx;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

public class TestExecutorC extends TestC {
	@Test
	void testExecutorC() {
		Executor.executeC(addIntPtrs, "addIntPtrs", addIntArgs);
		Executor.executeC(saxpy, "saxpy", saxpyArgs);
		
		String compiler = "gcc";
		
		Executor.executeC(addIntPtrs, "addIntPtrs", compiler, addIntArgs);
		Executor.executeC(saxpy, "saxpy", compiler, saxpyArgs);
		
		Options o1 = Options.createOptions();
		Options o2 = Options.createOptions();
		o2.insert("-Wall");
		o2.insert("-Wextra");
		o2.insert("-pedantic");
		
		Executor.executeC(addIntPtrs, "addIntPtrs", compiler, o1, addIntArgs);
		long eTime = Executor.executeC(saxpy, "saxpy", compiler, o2, saxpyArgs);
		
		assertTrue(eTime > 0);
	}
}
