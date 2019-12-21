package yacx;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.io.IOException;

import org.junit.jupiter.api.Test;

public class TestUtils {
	final String saxpy = "extern \"C\" __global__\n" + 
			"void saxpy(float a, float *x, float *y, float *out, size_t n) {\n" + 
			"  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;\n" + 
			"  if (tid < n) {\n" + 
			"    out[tid] = a * x[tid] + y[tid];\n" + 
			"  }\n" + 
			"}\n" + 
			"";

	@Test
	void loadFile() throws IOException {
		assertEquals(saxpy, Utils.loadFile("saxpy.cu"));
	}
}
