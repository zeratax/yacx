package yacx;

import org.junit.jupiter.api.BeforeAll;

public class TestC {
	static String addInts, saxpy;
	static KernelArg[] addIntArgs, saxpyArgs;
	static String[] addIntTypes, saxpyTypes;

	@BeforeAll
	static void initLibary() {
		// Load Libary
		Executor.loadLibrary();
	}

	@BeforeAll
	static void initCFunctions() {
		addInts = "void addInts(int32_t a, int32_t b, int32_t* result){\n"
				+ "    *result = a + b;\n"
				+ "}\n"
				+ "\n";

		saxpy = "#include <stddef.h>\n"
				+ "void saxpy(float a, float *x, float *y, float *out, int32_t n) {\n"
				+ "  for (int i = 0; i < n; i++) {\n"
				+ "    out[i] = a * x[i] + y[i];\n"
				+ "  }\n"
				+ "}";

		addIntArgs = new KernelArg[3];
		saxpyArgs = new KernelArg[5];

		addIntArgs[0] = IntArg.createValue(2);
		addIntArgs[1] = IntArg.createValue(3);
		addIntArgs[2] = IntArg.createOutput(1);

		saxpyArgs[0] = FloatArg.createValue(2f);
		saxpyArgs[1] = FloatArg.create(1f, 2f, 3.6f);
		saxpyArgs[2] = FloatArg.create(2f, 1f, 0f);
		saxpyArgs[3] = FloatArg.createOutput(3);
		saxpyArgs[4] = IntArg.createValue(3);

		addIntTypes = new String[] { "int32_t", "int32_t", "int32_t*" };
		saxpyTypes = new String[] { "float", "float*", "float*", "float*", "int32_t" };
	}
}
