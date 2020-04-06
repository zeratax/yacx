package yacx;

import java.util.ArrayList;
import java.util.List;

/**
 * Class to search a CUDA-capable device.
 */
public class Devices {
	/**
	 * Constructs a Device with the first CUDA capable device it finds.
	 * 
	 * @return a CUDA capable device
	 */
	public static native Device findDevice();

	/**
	 * Constructs a Device if a CUDA capable device with the identifier is
	 * available.
	 * 
	 * @param name name of the cuda device, e.g.'Tesla K20c'
	 * @return a CUDA capable device with the passed parameter as name
	 */
	public static native Device findDevice(String name);

	/**
	 * Constructs a Device if a CUDA capable device with the UUID is available.
	 * 
	 * @param uuid UUID of the cuda device,
	 *             e.g.'123e4567-e89b-12d3-a456-426655440000'
	 * @return a CUDA capable device with the passed parameter as UUID
	 * @throws ExecutorFailureException if CUDA version 9.1 or lower is used
	 */
	public static native Device findDeviceByUUID(String uuid) throws ExecutorFailureException;

	/**
	 * Returns a list of all available CUDA capable devices.
	 * 
	 * @return list of all CUDA capable devices
	 */
	public static native Device[] findDevices();

	/**
	 * Returns a list of all available CUDA capable devices which sufficient passes
	 * condition.
	 * 
	 * @param condition condition to filter devices
	 * 
	 * @return list of devices which sufficient passes condition
	 */
	public static List<Device> findDevices(DeviceCondition condition) {
		Device[] devices = findDevices();
		ArrayList<Device> filterd = new ArrayList<Device>(devices.length);

		for (Device device : devices) {
			if (condition.filterDevice(device))
				filterd.add(device);
		}

		return filterd;
	}

	/**
	 * Class for filter-condition for devices.
	 */
	public static abstract class DeviceCondition {
		/**
		 * Returns <code>true</code> if this device sufficient the filter-condition,
		 * <code>false</code> otherwise.
		 * 
		 * @param device Device which should checked
		 * @return <code>true</code> if this device sufficient the filter-condition,
		 *         <code>false</code> otherwise
		 */
		public abstract boolean filterDevice(Device device);
	}
}
