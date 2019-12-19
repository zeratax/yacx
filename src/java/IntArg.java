public class IntArg extends KernelArg {
	public final static int SIZE_BYTES = 4;

	public static native KernelArg createValue(int value);

	public static IntArg create(IntArg ints) {
		return create(ints, false);
	}


	public static native IntArg create(IntArg ints, boolean download);


	public static IntArg create(int ...ints) {
		return createInternal(ints, false);
	}

	public static IntArg create(int[] ints, boolean download) {
		assert(ints != null && ints.length > 0);

		return createInternal(ints, download);
	}

	private static native IntArg createInternal(int[] ints, boolean download);


	public static IntArg createOutput(int size) {
		assert(size > 0);

		return createOutputInternal(size);
	}

	private static native IntArg createOutputInternal(int length);

	IntArg(long handle) {
		super(handle);
	}

	public native int[] asIntArray();

	@Override
    public String toString(){
        return "IntArg " + super.toString();
    }
}
