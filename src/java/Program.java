public class Program  extends JNIHandle {
    public static native Program create(String kernelString, String kernelName);
    public native Kernel compile();
    public native Kernel compile(Options options);

    Program(long handle){
        super(handle);
    }
}