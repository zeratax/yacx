package yacx;

import org.junit.jupiter.api.BeforeAll;

public class TestC {
	static String addIntPtrs, saxpy;
	
	@BeforeAll
	static void initLibary() {
		//Load Libary
		Executor.loadLibary();
	}
	
	@BeforeAll
	static void initCFunctions() {
		addIntPtrs = "void addIntPtrs(int* a, int* b, int* result){\n" + 
				"    result = *a + *b;\n" + 
				"}\n" + 
				"\n";
		
		saxpy = "void saxpy(float* a, float *x, float *y, float *out, size_t* n) {\n" + 
				"  for (int i = 0; i < *n; i++) {\n" + 
				"    out[i] = *a * x[i] + y[i];\n" + 
				"  }\n" + 
				"}";
	}
}
