public class Program  extends JNIHandle {
    public static native Program create(String kernelString);

    public native Kernel kernel(String kernelName);

    Program(long handle){
        super(handle);
    }
}