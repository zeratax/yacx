package yacx;

/**
 * Class representing an double-array-argument for the kernel. <br>
 * Note: The size in bytes for one double-element is fixed and may differ from
 * the size of the corresponding data-type in CUDA, which is depending on your system.
 * So make sure the size of the corresponding data-type is matching {@link #SIZE_BYTES} to avoid
 * unexpected segmentation faults.
 */
public class DoubleArg extends ArrayArg {
	/**
	 * Size in bytes for one element.
	 */
	public final static long SIZE_BYTES = 1;

	/**
	 * Create a double-value-argument.
	 * @param value value of the argument
	 * @return a corresponding KernelArg representing this value
	 */
	public static native KernelArg createValue(double value);

	/**
	 * Create a new DoubleArg with the passed values. <br>
	 * This argument will be uploaded, but not be downloaded.
	 * @param doubles values for this argument
	 * @return a new DoubleArg with the passes values
	 */
	public static DoubleArg create(double ...doubles) {
		return create(doubles, false);
	}

	/**
	 * Create a new DoubleArg with the passed values.<br>
	 * This argument will be uploaded. The argument will be downloaded when
	 * <code>download</code> is <code>true</code>.
	 * @param doubles values for this argument
	 * @param download set whether the argument should be downloaded
	 * @return a new DoubleArg with the passes values
	 */
	public static DoubleArg create(double[] doubles, boolean download) {
		assert(doubles != null && doubles.length > 0);

		return createInternal(doubles, download);
	}

	private static native DoubleArg createInternal(double[] doubles, boolean download);

	/**
	 * Create an output-argument, which will be downloaded but not be uploaded. <br>
	 * This argument will be allocating enough memory for <code>length</code> double-elements.
	 * @param length number of elements for the output-argument
	 * @return a new DoubleArg
	 */
	public static DoubleArg createOutput(int length) {
		return new DoubleArg(createOutput(length * SIZE_BYTES));
	}

	DoubleArg(long handle) {
		super(handle);
	}

	/**
	 * Create a new array containing the data of this array.
	 * @return data of this array
	 */
	public native double[] asDoubleArray();
	
	@Override
	protected long getSizeBytes() {
		return SIZE_BYTES;
	}
	
	@Override
	public DoubleArg slice(int start, int end) {
		return new DoubleArg(slice(start * SIZE_BYTES, end * SIZE_BYTES));
	}

	@Override
    public String toString(){
        return "DoubleArg " + super.toString();
    }
}
