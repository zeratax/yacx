package yacx;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

public class TestExecutorC extends TestC {
	@Test
	void testExecutorC() {
		Executor.executeC(addInts, "addInts", addIntArgs);
		Executor.executeC(saxpy, "saxpy", saxpyArgs);

		assertEquals(5, ((IntArg) addIntArgs[2]).asIntArray()[0]);
		assertArrayEquals(new float[] { 2 + 2, 4 + 1, 7.2f }, ((FloatArg) saxpyArgs[3]).asFloatArray());

		String compiler = "gcc";

		Executor.executeC(addInts, "addInts", compiler, addIntArgs);
		Executor.executeC(saxpy, "saxpy", compiler, saxpyArgs);

		assertEquals(5, ((IntArg) addIntArgs[2]).asIntArray()[0]);
		assertArrayEquals(new float[] { 2 + 2, 4 + 1, 7.2f }, ((FloatArg) saxpyArgs[3]).asFloatArray());

		Options o1 = Options.createOptions();
		Options o2 = Options.createOptions();
		o2.insert("-Wall");
		o2.insert("-Wextra");
		o2.insert("-pedantic");

		Executor.executeC(addInts, "addInts", compiler, o1, addIntArgs);
		Executor.executeC(saxpy, "saxpy", compiler, o2, saxpyArgs);

		assertEquals(5, ((IntArg) addIntArgs[2]).asIntArray()[0]);
		assertArrayEquals(new float[] { 2 + 2, 4 + 1, 7.2f }, ((FloatArg) saxpyArgs[3]).asFloatArray());
	}
}
