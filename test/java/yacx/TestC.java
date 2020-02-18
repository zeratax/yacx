package yacx;

import org.junit.jupiter.api.BeforeAll;

public class TestC {
	static String addIntPtrs, saxpy;
	static KernelArg[] addIntArgs, saxpyArgs;

	@BeforeAll
	static void initLibary() {
		// Load Libary
		Executor.loadLibary();
	}

	@BeforeAll
	static void initCFunctions() {
		addIntPtrs = "void addIntPtrs(int* a, int* b, int* result){\n" + "    *result = *a + *b;\n" + "}\n" + "\n";

		saxpy = "#include <stddef.h>\n" + "void saxpy(float* a, float *x, float *y, float *out, int* n) {\n"
				+ "  for (int i = 0; i < *n; i++) {\n" + "    out[i] = *a * x[i] + y[i];\n" + "  }\n" + "}";

		addIntArgs = new KernelArg[3];
		saxpyArgs = new KernelArg[5];

		addIntArgs[0] = IntArg.create(2);
		addIntArgs[1] = IntArg.create(3);
		addIntArgs[2] = IntArg.createOutput(1);

		saxpyArgs[0] = FloatArg.create(2f);
		saxpyArgs[1] = FloatArg.create(1f, 2f, 3.6f);
		saxpyArgs[2] = FloatArg.create(2f, 1f, 0f);
		saxpyArgs[3] = FloatArg.createOutput(3);
		saxpyArgs[4] = IntArg.create(3);
	}
}
