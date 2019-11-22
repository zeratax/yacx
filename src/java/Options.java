public class Options extends JNIHandle{
    public native void insert(String op);

    public native void insert(String name, String value);

    Options(long handle) {
        super(handle);
    }
}