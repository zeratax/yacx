package yacx;

/**
 * Class representing an short-array-argument for the kernel. <br>
 * Note: The size in bytes for one short-element is fixed and may differ from
 * the size of the corresponding data-type in CUDA, which is depending on your system.
 * So make sure the size of the corresponding data-type is matching {@link #SIZE_BYTES} to avoid
 * unexpected segmentation faults.
 */
public class ShortArg extends ArrayArg {
	/**
	 * Size in bytes for one element.
	 */
	public final static long SIZE_BYTES = 1;

	/**
	 * Create a short-value-argument.
	 * @param value value of the argument
	 * @return a corresponding KernelArg representing this value
	 */
	public static native KernelArg createValue(short value);

	/**
	 * Create a new ShortArg with the passed values. <br>
	 * This argument will be uploaded, but not be downloaded.
	 * @param shorts values for this argument
	 * @return a new ShortArg with the passes values
	 */
	public static ShortArg create(short ...shorts) {
		return create(shorts, false);
	}

	/**
	 * Create a new ShortArg with the passed values.<br>
	 * This argument will be uploaded. The argument will be downloaded when
	 * <code>download</code> is <code>true</code>.
	 * @param shorts values for this argument
	 * @param download set whether the argument should be downloaded
	 * @return a new ShortArg with the passes values
	 */
	public static ShortArg create(short[] shorts, boolean download) {
		assert(shorts != null && shorts.length > 0);

		return createInternal(shorts, download);
	}

	private static native ShortArg createInternal(short[] shorts, boolean download);

	/**
	 * Create an output-argument, which will be downloaded but not be uploaded. <br>
	 * This argument will be allocating enough memory for <code>length</code> short-elements.
	 * @param length number of elements for the output-argument
	 * @return a new ShortArg
	 */
	public static ShortArg createOutput(int length) {
		return new ShortArg(createOutput(length * SIZE_BYTES));
	}

	ShortArg(long handle) {
		super(handle);
	}

	/**
	 * Create a new array containing the data of this array.
	 * @return data of this array
	 */
	public native short[] asShortArray();
	
	@Override
	protected long getSizeBytes() {
		return SIZE_BYTES;
	}
	
	@Override
	public ShortArg slice(int start, int end) {
		return new ShortArg(slice(start * SIZE_BYTES, end * SIZE_BYTES));
	}

	@Override
    public String toString(){
        return "ShortArg " + super.toString();
    }
}
