package yacx;

/**
 * Class representing an long-array-argument for the kernel. <br>
 * Note: The size in bytes for one long-element is fixed and may differ from
 * the size of the corresponding data-type in CUDA, which is depending on your system.
 * So make sure the size of the corresponding data-type is matching {@link #SIZE_BYTES} to avoid
 * unexpected segmentation faults.
 */
public class LongArg extends ArrayArg {
	/**
	 * Size in bytes for one element.
	 */
	public final static long SIZE_BYTES = 1;

	/**
	 * Create a long-value-argument.
	 * @param value value of the argument
	 * @return a corresponding KernelArg representing this value
	 */
	public static native KernelArg createValue(long value);

	/**
	 * Create a new LongArg with the passed values. <br>
	 * This argument will be uploaded, but not be downloaded.
	 * @param longs values for this argument
	 * @return a new LongArg with the passes values
	 */
	public static LongArg create(long ...longs) {
		return create(longs, false);
	}

	/**
	 * Create a new LongArg with the passed values.<br>
	 * This argument will be uploaded. The argument will be downloaded when
	 * <code>download</code> is <code>true</code>.
	 * @param longs values for this argument
	 * @param download set whether the argument should be downloaded
	 * @return a new LongArg with the passes values
	 */
	public static LongArg create(long[] longs, boolean download) {
		assert(longs != null && longs.length > 0);

		return createInternal(longs, download);
	}

	private static native LongArg createInternal(long[] longs, boolean download);

	/**
	 * Create an output-argument, which will be downloaded but not be uploaded. <br>
	 * This argument will be allocating enough memory for <code>length</code> long-elements.
	 * @param length number of elements for the output-argument
	 * @return a new LongArg
	 */
	public static LongArg createOutput(int length) {
		return new LongArg(createOutput(length * SIZE_BYTES));
	}

	LongArg(long handle) {
		super(handle);
	}

	/**
	 * Create a new array containing the data of this array.
	 * @return data of this array
	 */
	public native long[] asLongArray();
	
	@Override
	protected long getSizeBytes() {
		return SIZE_BYTES;
	}
	
	@Override
	public LongArg slice(int start, int end) {
		return new LongArg(slice(start * SIZE_BYTES, end * SIZE_BYTES));
	}

    @Override
    public String toString(){
        return "LongArg " + super.toString();
    }
}
