abstract public class JNIHandle {
    JNIHandle(long handle) {
        nativeHandle = handle;
    }

    public native void dispose();

    @SuppressWarnings({"unused", "FieldCanBeLocal"})
    private final long nativeHandle;

    @Override
    public String toString() {
        return String.format("0x%08X", nativeHandle);
    }
}
