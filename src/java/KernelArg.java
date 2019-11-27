public class KernelArg extends JNIHandle {
    public static native KernelArg create(float value);
    public static native KernelArg create(int value);
    public static native KernelArg create(double value);
    public static native KernelArg create(boolean value);

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
