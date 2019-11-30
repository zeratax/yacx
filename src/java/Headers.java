import java.io.File;

public class Headers extends JNIHandle {

    public static Headers createHeaders(String path){
        Headers h = createHeaders();
        h.insert(path);
        return h;
    }

    public static Headers createHeaders(String[] paths){
        Headers h = createHeaders();
        h.insert(paths);
        return h;
    }

    public static native Headers createHeaders();

    public void insert(String path){
        assert(fileExists(path));

        insertInternal(path);
    }

    private native void insertInternal(String path);

    public void insert(String[] paths){
        assert(fileExists(paths));

        insertInternal(paths);
    }

    private native void insertInternal(String[] paths);

    public native int getSize();

    public native String[] names();

    public native String[] content();


    private boolean fileExists(String path){
        return new File(path).exists();
    }

    private boolean fileExists(String[] paths){
        for (String path : paths){
            if (!(new File(path).exists()))
                return false;
        }

        return true;
    }

    Headers(long handle){
        super(handle);
    }
}