package yacx;

import java.util.Arrays;

/**
 * Class to help get a CUDA-capable device.
 */
public class Device extends JNIHandle {
	/**
	 * Constructs a Device with the first CUDA capable device it finds.
	 * @return a CUDA capable device
	 */
    public static native Device createDevice();

    /**
     * Constructs a Device if a CUDA capable device with the identifier is available
     * @param name name of the cuda device, e.g.'Tesla K20c'
     * @return a CUDA capable device with the passed parameter as name
     */
    public static Device createDevice(String name){
        assert(name != null && name.length() > 0);

        return createDeviceInternal(name);
    }

    private static native Device createDeviceInternal(String name);

    /**
     * Identifer string for the device.
     * @return identifer string
     */
    public native String getName();
    
    /**
     * Memory available on device for __constant__ variables in a CUDA C kernel in bytes.
     * @return memory in bytes
     */
    public native long getMemorySize();
    
    /**
     * Returns block with maximum dimension.
     * @return an array of integers with length 3 containing maximum block dimension for the first, second
     * and third dimension (x,y,z)
     */
    public native int[] getMaxBlock();
    
    /**
     * Returns grid with maximum dimension.
     * @return an array of integers with length 3 containing maximum grid dimension for the first, second
     * and third dimension (x,y,z)
     */
    public native int[] getMaxGrid();
    
    /**
     * Number of multiprocessors on device.
     * @return number of multiprocessors
     */
    public native int getMultiprocessorCount();
    
    /**
     * Peak clock frequency in kilohertz.
     * @return peak clock frequency
     */
    public native int getClockRate();
    
    /**
     * Peak memory clock frequency in kilohertz.
     * @return peak memory clock frequency
     */
    public native int getMemoryClockRate();
    
    /**
     * Global memory bus width in bits.
     * @return bus width
     */
    public native int getBusWidth();
    
    /**
     * Minor compute capability version number.
     * @return version number
     */
    public native int getMinorVersion();
    
    /**
     * Major compute capability version number.
     * @return version number
     */
    public native int getMajorVersion();

    /**
	 * Create a new Device.
	 * @param handle Pointer to corresponding C-Device-Object
	 */
    Device(long handle){
        super(handle);
    }

    @Override
    public String toString(){
        return "Device: " + getName() + " (Memory: " + getMemorySize()/1024/1024 + " MebiBytes, Blocks: " + Arrays.toString(getMaxBlock())
            + ", Grids: " + Arrays.toString(getMaxGrid()) + ")";
    }
}
