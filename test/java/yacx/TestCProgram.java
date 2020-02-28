package yacx;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestMethodOrder;
import org.junit.jupiter.api.MethodOrderer.OrderAnnotation;

@TestMethodOrder(OrderAnnotation.class)
public class TestCProgram extends TestC {
	static String addIntsInvalid = "void addInts(int32_t a, int32_t b, int32_t* result){\n"
			+ "    *result = a + c;\n"
			+ "}";

	static CProgram addIntC, saxpyC;

	@Test
	@Order(1)
	void testGetTypes() {
		String[] types = CProgram.getTypes(addIntArgs);
		assertEquals(types.length, 3);
		assertEquals("int32_t", types[0]);
		assertEquals("int32_t", types[1]);
		assertTrue(types[2].endsWith("*"));

		types = CProgram.getTypes(saxpyArgs);
		assertEquals(types.length, 5);
		assertEquals("float", types[0]);
		assertTrue(types[1].endsWith("*"));
		assertTrue(types[2].endsWith("*"));
		assertTrue(types[3].endsWith("*"));
		assertEquals("int32_t", types[4]);
	}

	@Test
	@Order(2)
	void testCompileInvalid() {
		assertThrows(NullPointerException.class, () -> {
			CProgram.create(null, "addInts", addIntTypes);
		});

		assertThrows(NullPointerException.class, () -> {
			CProgram.create(addInts, null, addIntTypes);
		});

		assertThrows(IllegalArgumentException.class, () -> {
			CProgram.create(addInts, "addInts", new String[0]);
		});

		assertThrows(ExecutorFailureException.class, () -> {
			CProgram.create(addIntsInvalid, "addInts", addIntTypes);
		});
	}

	@Test
	@Order(3)
	void testCompile() {
		addIntC = CProgram.create(addInts, "addInts", addIntTypes);

		String compiler = "gcc";
		Options options = Options.createOptions();
		options.insert("-Wall");
		options.insert("-pedantic");
		;

		saxpyC = CProgram.create(saxpy, "saxpy", saxpyTypes, compiler);
		saxpyC = CProgram.create(saxpy, "saxpy", saxpyTypes, compiler, options);
	}

	@Test
	@Order(4)
	void testExecuteInvalid() {
		assertThrows(IllegalArgumentException.class, () -> {
			addIntC.execute();
		});

		assertThrows(ExecutorFailureException.class, () -> {
			saxpyC.execute(saxpyArgs[0], saxpyArgs[1], saxpyArgs[2], saxpyArgs[3]);
		});

		assertThrows(ExecutorFailureException.class, () -> {
			addIntC.execute(addIntArgs[0], addIntArgs[1], addIntArgs[2], addIntArgs[2]);
		});
	}

	@Test
	@Order(5)
	void testExecute() {
		addIntC.execute(addIntArgs);
		assertEquals(5, ((IntArg) addIntArgs[2]).asIntArray()[0]);

		saxpyC.execute(saxpyArgs);
		assertArrayEquals(new float[] { 2 + 2, 4 + 1, 7.2f }, ((FloatArg) saxpyArgs[3]).asFloatArray());
	}
}
