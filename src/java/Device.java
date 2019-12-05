import java.util.Arrays;

public class Device extends JNIHandle {
    public static native Device createDevice();

    public static Device createDevice(String name){
        assert(name != null && name.length() > 0);

        return createDeviceInternal(name);
    }

    private static native Device createDeviceInternal(String name);

    public native String getName();
    public native long getMemorySize();
    public native int[] getMaxBlock();
    public native int[] getMaxGrid();
    public native int getMultiprocessorCount();
    public native int getClockRate();
    public native int getMemoryClockRate();
    public native int getBusWidth();

    Device(long handle){
        super(handle);
    }

    @Override
    public String toString(){
        return "Device: " + getName() + " (Memory: " + getMemorySize()/1024/1024 + " MebiBytes, Blocks: " + Arrays.toString(getMaxBlock())
            + ", Grids: " + Arrays.toString(getMaxGrid()) + ")";
    }
}