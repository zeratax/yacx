package yacx;

import java.util.Arrays;

/**
 * Class to handle a CUDA-capable device.
 */
public class Device extends JNIHandle {
	/**
	 * Identifer string for the device.
	 * 
	 * @return identifer string
	 */
	public native String getName();

	/**
	 * Memory available on device for __constant__ variables in a CUDA C kernel in
	 * bytes.
	 * 
	 * @return memory in bytes
	 */
	public native long getMemorySize();

	/**
	 * Returns block with maximum dimension.
	 * 
	 * @return an array of integers with length 3 containing maximum block dimension
	 *         for the first, second and third dimension (x,y,z)
	 */
	public native int[] getMaxBlock();

	/**
	 * Returns grid with maximum dimension.
	 * 
	 * @return an array of integers with length 3 containing maximum grid dimension
	 *         for the first, second and third dimension (x,y,z)
	 */
	public native int[] getMaxGrid();

	/**
	 * Number of multiprocessors on device.
	 * 
	 * @return number of multiprocessors
	 */
	public native int getMultiprocessorCount();

	/**
	 * Peak clock frequency in kilohertz.
	 * 
	 * @return peak clock frequency
	 */
	public native int getClockRate();

	/**
	 * Peak memory clock frequency in kilohertz.
	 * 
	 * @return peak memory clock frequency
	 */
	public native int getMemoryClockRate();

	/**
	 * Global memory bus width in bits.
	 * 
	 * @return bus width
	 */
	public native int getBusWidth();

	/**
	 * Minor compute capability version number.
	 * 
	 * @return version number
	 */
	public native int getMinorVersion();

	/**
	 * Major compute capability version number.
	 * 
	 * @return version number
	 */
	public native int getMajorVersion();

	/**
	 * Returns the UUID for this device or <code>null</code> if not available (CUDA
	 * version 9.2 or higher required).
	 * 
	 * @return 16-byte UUID of the device as hexadecimal string or <code>null</code>
	 */
	public native String getUUID();

	/**
	 * Create a new Device.
	 * 
	 * @param handle Pointer to corresponding C-Device-Object
	 */
	Device(long handle) {
		super(handle);
	}

	@Override
	public String toString() {
		String uuid = getUUID();

		if (uuid == null) {
			return "Device: " + getName() + " (Memory: " + getMemorySize() / 1024 / 1024 + " MB, Blocks: "
					+ Arrays.toString(getMaxBlock()) + ", Grids: " + Arrays.toString(getMaxGrid())
					+ ", computeversions: " + getMinorVersion() + "-" + getMajorVersion() + ")";
		} else {
			return "Device: " + getName() + " (UUID: " + uuid + " ,Memory: " + getMemorySize() / 1024 / 1024
					+ " MB, Blocks: " + Arrays.toString(getMaxBlock()) + ", Grids: " + Arrays.toString(getMaxGrid())
					+ ", computeversions: " + getMinorVersion() + "-" + getMajorVersion() + ")";
		}
	}
	
	//Devices are Singeltons and should be not destroyed
	@Override
	public void dispose() {}
}
