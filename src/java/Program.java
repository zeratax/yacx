public class Program extends JNIHandle {

    public static Program create(String kernelString, String kernelName) {
        assert(kernelString != null && kernelString.length() > 0);
        assert(kernelName != null && kernelName.length() > 0);

        return createInternal(kernelString, kernelName);
    }

    private static native Program createInternal(String kernelString, String kernelName);

    public static Program create(String kernelString, String kernelName, Headers headers) {
        assert(kernelString != null && kernelString.length() > 0);
        assert(kernelName != null && kernelName.length() > 0);
        assert(headers != null);

        return createInternal(kernelString, kernelName, headers);
    }

    private static native Program createInternal(String kernelString, String kernelName, Headers headers);
    
    public Program instantiate(String ...templateParameter) {
    	assert(templateParameter != null && templateParameter.length > 0);
    	
    	for (int i = 0; i < templateParameter.length; i++) {
    		assert(templateParameter[i] != null);
    		
    		instantiateInternal(templateParameter[i]);
    	}
    	
    	return this;
    }
    
    private native void instantiateInternal(String templateParameter);

    public native Kernel compile();

    public Kernel compile(Options options) {
        assert(options != null);

        return compileInternal(options);
    }

    private native Kernel compileInternal(Options options);

    Program(long handle) {
        super(handle);
    }
}