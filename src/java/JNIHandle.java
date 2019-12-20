package yacx;

abstract public class JNIHandle {
    JNIHandle(long handle) {
        nativeHandle = handle;
    }

    public native void dispose();

    private final long nativeHandle;

    @Override
    public boolean equals(Object o) {
    	return o != null && o instanceof JNIHandle && nativeHandle == ((JNIHandle) o).nativeHandle;
    }

    @Override
    public String toString() {
        return String.format("0x%08X", nativeHandle);
    }
}
