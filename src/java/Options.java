public class Options extends JNIHandle{
    public native void insert(String op);

    public native void insert(String name, String value);

    public native String options();

    Options(long handle) {
        super(handle);
    }
}