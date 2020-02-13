package yacx;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.Ignore;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestMethodOrder;
import org.junit.jupiter.api.MethodOrderer.OrderAnnotation;

@Ignore
@TestMethodOrder(OrderAnnotation.class)
public class TestCProgram extends TestC {
	static String addIntPtrsInvalid = "void addIntPtrs(int* a, int* b, int* result){\n" + 
			"    result = *a + *c;\n" + 
			"}";
	
	static CProgram addIntC, saxpyC;
	static KernelArg[] addIntArgs, saxpyArgs;
	
	@BeforeAll
	static void initArgs() {
		addIntArgs = new KernelArg[3];
		saxpyArgs = new KernelArg[5];
		
		addIntArgs[0] = IntArg.create(2);
		addIntArgs[1] = IntArg.create(3);
		addIntArgs[2] = IntArg.createOutput(1);
		
		saxpyArgs[0] = FloatArg.create(2f);
		saxpyArgs[1] = FloatArg.create(1f, 2f, 3.6f);
		saxpyArgs[2] = FloatArg.create(2f, 1f, 0f);
		saxpyArgs[3] = FloatArg.createOutput(3);
		saxpyArgs[4] = IntArg.createOutput(3);
	}
	
	@Test
	@Order(1)
	void testCompileInvalid() {
		assertThrows(NullPointerException.class, () -> {
			CProgram.create(null, "addIntPtrs", 3);
		});
		
		assertThrows(NullPointerException.class, () -> {
			CProgram.create(addIntPtrs, null, 3);
		});
		
		assertThrows(NullPointerException.class, () -> {
			CProgram.create(addIntPtrs, "addIntPtrs", -17);
		}); 
		
		assertThrows(NullPointerException.class, () -> {
			CProgram.create(addIntPtrs, "addIntPtrs", 0);
		}); 
		
		assertThrows(ExecutorFailureException.class, () -> {
			CProgram.create(addIntPtrsInvalid, "addIntPtrs", 3);
		}); 
	}
	
	@Test
	@Order(2)
	void testCompile() {
		addIntC = CProgram.create(addIntPtrs, "addIntPtrs", 3);
		
		String compiler = "gcc";
		Options options = Options.createOptions();
		options.insert("-Wall");
		options.insert("-pedantic");;
		
		saxpyC = CProgram.create(saxpy, "saxpy", 5, compiler);
		saxpyC = CProgram.create(saxpy, "saxpy", 3, compiler, options);
	}
	
	@Test
	@Order(3)
	void testExecuteInvalid() {
		assertThrows(IllegalArgumentException.class, () -> {
			addIntC.execute();
		});
		
		assertThrows(IllegalArgumentException.class, () -> {
			saxpyC.execute(saxpyArgs[0], saxpyArgs[1], saxpyArgs[2], saxpyArgs[3]);
		});
		
		assertThrows(IllegalArgumentException.class, () -> {
			addIntC.execute(addIntArgs[0], addIntArgs[1], addIntArgs[2], addIntArgs[2]);
		});
	}
	
	@Test
	@Order(4)
	void testExecute() {
		addIntC.execute(addIntArgs);
		assertEquals(5, ((IntArg) addIntArgs[2]).asIntArray()[0]);
		
		saxpyC.execute(saxpyArgs);
		assertArrayEquals(new float[] {2+2, 4+1, 7.2f}, ((FloatArg) addIntArgs[3]).asFloatArray()); 
	}
}
