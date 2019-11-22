
public class Program extends JNIHandle {
    public static native Program create(String kernelCode, String kernelName, String buildOptions);
    public native void compile(Options options);
    public native void launch(KernelArg[] args);

    Program(long handle) {

        super(handle);
    }

    public native void build();
}
