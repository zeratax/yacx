package yacx;

import static org.junit.jupiter.api.Assertions.*;

import java.util.List;

import org.junit.jupiter.api.Test;

class TestDevices extends TestJNI {

	@Test
	void test() {
		// Find a Device-Object
		Device device = Devices.findDevice();

		assertNotNull(device);

		// Get name from device
		final String name = device.getName();

		assertNotNull(name);

		// Find a device with equal name
		Device device2 = Devices.findDevice(name);

		assertNotNull(device2);
		assertEquals(name, device2.getName());

		// Should be same device, cause devices are singletons
		assertEquals(device, device2);

		// Find all devices
		Device[] devices = Devices.findDevices();

		assertTrue(devices.length > 0);

		// Device should be in devices-array
		assertTrue(contains(devices, device));
		
		// Find device by name using filter-method
		List<Device> deviceList = Devices.findDevices(new Devices.DeviceCondition() {
			
			@Override
			public boolean filterDevice(Device device) {
				return device.getName().equals(name);
			}
		});
		
		assertEquals(1, deviceList.size());
		assertEquals(device, deviceList.get(0));
	}

	/**
	 * Returns <code>true</code> if device is in devices-array, <code>false</code>
	 * otherwise.
	 * 
	 * @param devices array of devices
	 * @param device  device which should be searched
	 * @return <code>true</code> if device was found, <code>false</code> otherwise
	 */
	boolean contains(Device[] devices, Device device) {
		for (Device d : devices) {
			if (d.equals(device))
				return true;
		}

		return false;
	}
}
