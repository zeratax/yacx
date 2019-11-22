
public class Kernel extends JNIHandle {
    public static native Kernel create(String kernelCode, String kernelName, String buildOptions);
    public native void compile(Options options);
    public native void launch(KernelArg[] args);

    Kernel(long handle) {
        super(handle);
    }

    public native void build();
}
