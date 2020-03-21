package yacx;

import org.junit.jupiter.api.BeforeAll;

public class TestJNI {
	@BeforeAll
	static void initLibary() {
		// Load Libary
		Executor.loadLibrary();
	}

	@BeforeAll
	static void disableAssertions() {
		// Disable Assertions
		ClassLoader.getSystemClassLoader().setDefaultAssertionStatus(false);
	}
}
