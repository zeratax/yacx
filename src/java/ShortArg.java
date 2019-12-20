public class ShortArg extends KernelArg {
	public final static int SIZE_BYTES = 2;

	public static native KernelArg createValue(short value);

	public static ShortArg create(ShortArg shorts) {
		return create(shorts, false);
	}

	public static native ShortArg create(ShortArg shorts, boolean download);


	public static ShortArg create(short ...shorts) {
		return createInternal(shorts, false);
	}

	public static ShortArg create(short[] shorts, boolean download) {
		assert(shorts != null && shorts.length > 0);

		return createInternal(shorts, download);
	}

	private static native ShortArg createInternal(short[] shorts, boolean download);


	public static ShortArg createOutput(int size) {
		assert(size > 0);

		return createOutputInternal(size);
	}

	private static native ShortArg createOutputInternal(int length);

	ShortArg(long handle) {
		super(handle);
	}

	public native short[] asShortArray();

	@Override
    public String toString(){
        return "ShortArg " + super.toString();
    }
}
