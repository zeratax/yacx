package yacx;

/**
 * Class representing an int-array-argument for the kernel. <br>
 * Note: The size in bytes for one int-element is fixed and may differ from
 * the size of the corresponding data-type in CUDA, which is depending on your system.
 * So make sure the size of the corresponding data-type is matching {@link #SIZE_BYTES} to avoid
 * unexpected segmentation faults.
 */
public class IntArg extends ArrayArg {
	/**
	 * Size in bytes for one element.
	 */
	public final static long SIZE_BYTES = 4;

	/**
	 * Create a int-value-argument.
	 * @param value value of the argument
	 * @return a corresponding KernelArg representing this value
	 */
	public static native KernelArg createValue(int value);

	/**
	 * Create a new IntArg with the passed values. <br>
	 * This argument will be uploaded, but not be downloaded.
	 * @param ints values for this argument
	 * @return a new IntArg with the passes values
	 */
	public static IntArg create(int ...ints) {
		return create(ints, false);
	}

	/**
	 * Create a new IntArg with the passed values.<br>
	 * This argument will be uploaded. The argument will be downloaded when
	 * <code>download</code> is <code>true</code>.
	 * @param ints values for this argument
	 * @param download set whether the argument should be downloaded
	 * @return a new IntArg with the passes values
	 */
	public static IntArg create(int[] ints, boolean download) {
		assert(ints != null && ints.length > 0);

		return createInternal(ints, download);
	}

	private static native IntArg createInternal(int[] ints, boolean download);

	/**
	 * Create an output-argument, which will be downloaded but not be uploaded. <br>
	 * This argument will be allocating enough memory for <code>length</code> int-elements.
	 * @param length number of elements for the output-argument
	 * @return a new IntArg
	 */
	public static IntArg createOutput(int length) {
		return new IntArg(createOutput(length * SIZE_BYTES));
	}

	IntArg(long handle) {
		super(handle);
	}

	/**
	 * Create a new array containing the data of this array.
	 * @return data of this array
	 */
	public native int[] asIntArray();
	
	@Override
	protected long getSizeBytes() {
		return SIZE_BYTES;
	}
	
	@Override
	public IntArg slice(int start, int end) {
		return new IntArg(slice(start * SIZE_BYTES, end * SIZE_BYTES + SIZE_BYTES));
	}

	@Override
    public String toString(){
        return "IntArg " + super.toString();
    }
}
