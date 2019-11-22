

public class KernelArg extends JNIHandle {
    public static native KernelArg create(float array);
    public static native KernelArg create(int array);
    public static native KernelArg create(double array);
    public static native KernelArg create(boolean array);

    public static native KernelArg create(float[] array, boolean output);
    public static native KernelArg create(int[] array, boolean output);
    public static native KernelArg create(double[] array, boolean output);
    public static native KernelArg create(boolean[] array, boolean output);

    public static native KernelArg createOutput(long size);

    KernelArg(long handle) {
        super(handle);
    }

    public native float[] asFloatArray();
    public native int[] asIntArray();
    public native double[] asDoubleArray();
    public native boolean[] asBooleanArray();
}
