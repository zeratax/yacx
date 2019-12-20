package yacx;

public class FloatArg extends KernelArg {
	public final static int SIZE_BYTES = 4;

	public static native KernelArg createValue(float value);

	public static FloatArg create(float ...floats) {
		return createInternal(floats, false);
	}

	public static FloatArg create(float[] floats, boolean download) {
		assert(floats != null && floats.length > 0);

		return createInternal(floats, download);
	}

	private static native FloatArg createInternal(float[] floats, boolean download);


	public static FloatArg createOutput(int size) {
		assert(size > 0);

		return createOutputInternal(size);
	}

	private static native FloatArg createOutputInternal(int length);

	FloatArg(long handle) {
		super(handle);
	}

	public native float[] asFloatArray();

	@Override
    public String toString(){
        return "FloatArg " + super.toString();
    }
}
