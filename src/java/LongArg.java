package yacx;

public class LongArg extends KernelArg {
	public final static int SIZE_BYTES = 8;

	public static native KernelArg createValue(long value);

	public static LongArg create(LongArg longs){
		return create(longs, false);
	}

	public static native LongArg create(LongArg longs, boolean download);


	public static LongArg create(long ...longs) {
		return createInternal(longs, false);
	}

	public static LongArg create(long[] longs, boolean download) {
		assert(longs != null && longs.length > 0);

		return createInternal(longs, download);
	}

	private static native LongArg createInternal(long[] longs, boolean download);


	public static LongArg createOutput(int size) {
		assert(size > 0);

		return createOutputInternal(size);
	}

	private static native LongArg createOutputInternal(int length);

	public native long[] asLongArray();

	LongArg(long handle) {
		super(handle);
	}

    @Override
    public String toString(){
        return "LongArg " + super.toString();
    }
}
