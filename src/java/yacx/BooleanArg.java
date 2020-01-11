package yacx;

/**
 * Class representing an boolean-array-argument for the kernel. <br>
 * Note: The size in bytes for one boolean-element is fixed and may differ from
 * the size of the corresponding data-type in CUDA, which is depending on your system.
 * So make sure the size of the corresponding data-type is matching {@link #SIZE_BYTES} to avoid
 * unexpected segmentation faults.
 */
public class BooleanArg extends ArrayArg {
	/**
	 * Size in bytes for one element.
	 */
	public final static long SIZE_BYTES = 1;

	/**
	 * Create a boolean-value-argument.
	 * @param value value of the argument
	 * @return a corresponding KernelArg representing this value
	 */
	public static native KernelArg createValue(boolean value);

	/**
	 * Create a new BooleanArg with the passed values. <br>
	 * This argument will be uploaded, but not be downloaded.
	 * @param booleans values for this argument
	 * @return a new BooleanArg with the passes values
	 */
	public static BooleanArg create(boolean ...booleans) {
		return create(booleans, false);
	}

	/**
	 * Create a new BooleanArg with the passed values.<br>
	 * This argument will be uploaded. The argument will be downloaded when
	 * <code>download</code> is <code>true</code>.
	 * @param booleans values for this argument
	 * @param download set whether the argument should be downloaded
	 * @return a new BooleanArg with the passes values
	 */
	public static BooleanArg create(boolean[] booleans, boolean download) {
		assert(booleans != null && booleans.length > 0);

		return createInternal(booleans, download);
	}

	private static native BooleanArg createInternal(boolean[] booleans, boolean download);

	/**
	 * Create an output-argument, which will be downloaded but not be uploaded. <br>
	 * This argument will be allocating enough memory for <code>length</code> boolean-elements.
	 * @param length number of elements for the output-argument
	 * @return a new BooleanArg
	 */
	public static BooleanArg createOutput(int length) {
		return new BooleanArg(createOutput(length * SIZE_BYTES));
	}

	BooleanArg(long handle) {
		super(handle);
	}

	/**
	 * Create a new array containing the data of this array.
	 * @return data of this array
	 */
	public native boolean[] asBooleanArray();
	
	@Override
	protected long getSizeBytes() {
		return SIZE_BYTES;
	}
	
	@Override
	public BooleanArg slice(int start, int end) {
		return new BooleanArg(slice(start * SIZE_BYTES, end * SIZE_BYTES + SIZE_BYTES));
	}

	@Override
    public String toString(){
        return "BooleanArg " + super.toString();
    }
}
