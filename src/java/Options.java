import java.util.Arrays;

public class Options extends JNIHandle {

    public static Options createOptions(String ...options) {
        Options o = createOptions();
        for (String option : options)
        	o.insert(option);
        return o;
    }

    private static native Options createOptions();

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

    public native String[] getOptions();

    Options(long handle) {
        super(handle);
    }

    @Override
    public String toString(){
        return "Options " + Arrays.toString(getOptions());
    }
}