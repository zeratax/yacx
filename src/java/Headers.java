package yacx;

import java.io.File;

public class Headers extends JNIHandle {

    public static Headers createHeaders(String ...paths) {
        Headers h = createHeaders();
        h.insert(paths);
        return h;
    }

    public static native Headers createHeaders();

    public void insert(String ...paths) {
    	assert(paths != null && paths.length > 0);
        assert(fileExists(paths));

        insertInternal(paths);
    }

    private native void insertInternal(String[] paths);

    public native int getSize();

    public native String[] names();

    public native String[] content();

    private boolean fileExists(String[] paths) {
        for (String path : paths) {
        	assert(path != null);
            if (!(new File(path).exists()))
                return false;
        }

        return true;
    }

    Headers(long handle) {
        super(handle);
    }

    @Override
    public String toString(){
        return "Headers " + names();
    }
}
