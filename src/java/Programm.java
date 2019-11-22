public class Programm  extends JNIHandle {
    public static native Programm create(String kernelString);

    public native Program program(String kernelName);


    Programm(long handle){
        super(handle);
    }
}