public class Program  extends JNIHandle {
    public static native Program create(String kernelString, String kernelName);
    public native Kernel compile();

    Program(long handle){
        super(handle);
    }
}