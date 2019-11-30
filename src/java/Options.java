public class Options extends JNIHandle {

    public static Options createOptions(String option) {
        Options o = createOptions();
        o.insert(option);
        return o;
    }

    public static native Options createOptions();

    public void insert(String option) {
        assert(option != null && option.length() > 0);

        insertInternal(option);
    }

    private native void insertInternal(String option);

    public void insert(String name, Object value) {
        assert(name != null && name.length() > 0);
        assert(value != null && value.toString().length() > 0);

        insertInternal(name, value.toString());
    }

    private native void insertInternal(String name, String value);

    public native int getSize();

    public native String[] options();

    Options(long handle) {
        super(handle);
    }
}