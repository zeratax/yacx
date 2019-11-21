package opencl.executor;

public class Executor {
    private static void loadLibrary()
    {
        System.loadLibrary("cudaexecutor-jni");
    }

    private native static void initExecutor();

    public static void init(){
        loadLibrary();
        initExecutor();
    }

    public native static double execute(Kernel kernel, KernelArg[] args);

}
