package yacx;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

class TestDevice extends TestJNI {

	@Test
	void test() {
		// Find a Device-Object
		Device device = Devices.findDevice();

		// Check functions should be return meaningful values (and no
		// UnsatisfiedLinkError)
		assertNotNull(device.getName());
		assertTrue(device.getMemorySize() > 0);
		assertTrue(device.getMaxBlock().length == 3);
		assertTrue(device.getMaxGrid().length == 3);
		for (int i = 0; i < 3; i++) {
			assertTrue(device.getMaxBlock()[i] > 0);
			assertTrue(device.getMaxGrid()[i] > 0);
		}
		assertTrue(device.getMultiprocessorCount() > 0);
		assertTrue(device.getClockRate() > 0);
		assertTrue(device.getMemoryClockRate() > 0);
		assertTrue(device.getBusWidth() > 0);
		assertTrue(device.getMinorVersion() >= 0);
		assertTrue(device.getMajorVersion() > 0);
		assertTrue(device.getSharedMemPerBlock() >= 0);
		assertTrue(device.getSharedMemPerMultiprocessor() >= 0);
		// This could be null
		device.getUUID();

		// For Debug
		System.out.println("Device-Information:");
		System.out.println(device);
	}

}
