public class Programm  extends JNIHandle {
    public static native Programm create(String kernelString);

    public native Kernel kernel(String kernelName);

    Programm(long handle){
        super(handle);
    }
}