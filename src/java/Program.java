package yacx;

public class Program extends JNIHandle {

    public static Program create(String kernelString, String kernelName) {
        assert(kernelString != null && kernelString.length() > 0);
        assert(kernelName != null && kernelName.length() > 0);

        return createInternal(kernelString, kernelName);
    }

    private static native Program createInternal(String kernelString, String kernelName);

    public static Program create(String kernelString, String kernelName, Headers headers) {
        assert(kernelString != null && kernelString.length() > 0);
        assert(kernelName != null && kernelName.length() > 0);
        assert(headers != null);

        return createInternal(kernelString, kernelName, headers);
    }

    private static native Program createInternal(String kernelString, String kernelName, Headers headers);

    public native Kernel compile();

    public Kernel compile(Options options) {
        assert(options != null);

        return compileInternal(options);
    }

    private native Kernel compileInternal(Options options);

    Program(long handle) {
        super(handle);
    }
}
