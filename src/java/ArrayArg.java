public class ArrayArg extends KernelArg {
    private byte type;

    private final static byte FLOAT = 1, INT = 2, LONG = 3, DOUBLE = 4, BOOLEAN = 5;

    public static ArrayArg create(float[] array, boolean output) {
        assert(array != null && array.length > 0);

        ArrayArg arg = createInternal(array, output);
        arg.type = FLOAT;

        return arg;
    }

    public static ArrayArg create(int[] array, boolean output) {
        assert(array != null && array.length > 0);

        ArrayArg arg = createInternal(array, output);
        arg.type = INT;

        return arg;
    }

    public static ArrayArg create(long[] array, boolean output) {
        assert(array != null && array.length > 0);

        ArrayArg arg = createInternal(array, output);
        arg.type = LONG;

        return arg;
    }

    public static ArrayArg create(double[] array, boolean output) {
        assert(array != null && array.length > 0);

        ArrayArg arg = createInternal(array, output);
        arg.type = DOUBLE;

        return arg;
    }

    public static ArrayArg create(boolean[] array, boolean output) {
        assert(array != null && array.length > 0);

        ArrayArg arg = createInternal(array, output);
        arg.type = BOOLEAN;

        return arg;
    }

    private static native ArrayArg createInternal(float[] array, boolean output);
    private static native ArrayArg createInternal(int[] array, boolean output);
    private static native ArrayArg createInternal(long[] array, boolean output);
    private static native ArrayArg createInternal(double[] array, boolean output);
    private static native ArrayArg createInternal(boolean[] array, boolean output);

    public static ArrayArg createOutput(long size) {
        assert(size > 0);

        return createOutputInternal(size);
    }

    private static native ArrayArg createOutputInternal(long size);

    public float[] asFloatArray() {
        assert(type == FLOAT || type == 0);

        return asFloatArrayInternal();
    }

    public int[] asIntArray() {
        assert(type == INT || type == 0);

        return asIntArrayInternal();
    }

    public long[] asLongArray() {
        assert(type == LONG || type == 0);

        return asLongArrayInternal();
    }

    public double[] asDoubleArray() {
        assert(type == DOUBLE || type == 0);

        return asDoubleArrayInternal();
    }

    public boolean[] asBooleanArray() {
        assert(type == BOOLEAN || type == 0);

        return asBooleanArrayInternal();
    }

    private native float[] asFloatArrayInternal();
    private native int[] asIntArrayInternal();
    private native long[] asLongArrayInternal();
    private native double[] asDoubleArrayInternal();
    private native boolean[] asBooleanArrayInternal();

    ArrayArg(long handle) {
        super(handle);
    }

    @Override
    public String toString() {
        return "ArrayArg (Type: " + type + ") " + super.toString();
    }
}