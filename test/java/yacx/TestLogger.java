package yacx;

import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.io.File;
import java.io.IOException;

import org.junit.jupiter.api.Test;

import yacx.Logger.LogLevel;

public class TestLogger extends TestJNI {
	static final File logFile = new File("logfile");

	@Test
	void test() throws IOException {
		if (logFile.exists())
			if (!logFile.delete())
				throw new IOException("could not delete " + logFile.getAbsolutePath());

		Logger.setCout(false);
		Logger.setCerr(true);
		Logger.setLogfile(logFile.getAbsolutePath());

		// Invalid kernel (very simalar to saxpy-kernel)
		String kernelInvalid = "extern \"C\" __global__\n"
				+ "void saxpy(float a, float *x, float *y, float *out, size_t n) {\n"
				+ "  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;\n" + "  if (tid < m) {\n" + // <- There is a m,
																										// not a n. It
																										// is not
																										// compilable!
				"    out[tid] = a * x[tid] + y[tid];\n" + "  }\n" + "}\n" + "";

		assertThrows(ExecutorFailureException.class, () -> {
			Executor.launch(kernelInvalid, "saxpy", 1, 1, FloatArg.createValue(1f));
		});

		// Something should be logged
		String log = Utils.loadFile(logFile.getAbsolutePath());

		assertNotNull(log);
		assertTrue(log.length() > 0);

		// More verbose loglevel
		Logger.setLogLevel(LogLevel.DEBUG);

		assertThrows(ExecutorFailureException.class, () -> {
			Executor.launch(kernelInvalid, "saxpy", 1, 1, FloatArg.createValue(1f));
		});

		String log2 = Utils.loadFile(logFile.getAbsolutePath());

		assertNotNull(log2);
		assertTrue(log2.length() > log.length());
	}
}
