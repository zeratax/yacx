package opencl.executor;

public class Executor {
    public static void loadLibrary()
    {
        System.loadLibrary("cudaexecutor-jni");
    }

    public native static double execute(Kernel kernel, KernelArg[] args);

}
