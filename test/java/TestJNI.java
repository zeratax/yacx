package yacx;

import java.io.IOException;

import org.junit.jupiter.api.BeforeAll;

public class TestJNI {
	static String saxpy, filterk;
	
	@BeforeAll
	static void initLibary() {
		//Load Libary
		Executor.loadLibary();
	}
	
	@BeforeAll
	static void loadKernelStrings() throws IOException {
		//Load Saxpy and Filter-Kernel as String
		saxpy = Utils.loadFile("saxpy.cu");
		filterk = Utils.loadFile("filter_k.cu");
	}
	
	@BeforeAll
	static void disableAssertions() {
		//Disable Assertions
		ClassLoader.getSystemClassLoader().setDefaultAssertionStatus(false);
	}
}
