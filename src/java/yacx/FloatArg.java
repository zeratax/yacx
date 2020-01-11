package yacx;

/**
 * Class representing an float-array-argument for the kernel. <br>
 * Note: The size in bytes for one float-element is fixed and may differ from
 * the size of the corresponding data-type in CUDA, which is depending on your system.
 * So make sure the size of the corresponding data-type is matching {@link #SIZE_BYTES} to avoid
 * unexpected segmentation faults.
 */
public class FloatArg extends ArrayArg {
	/**
	 * Size in bytes for one element.
	 */
	public final static long SIZE_BYTES = 4;

	/**
	 * Create a float-value-argument.
	 * @param value value of the argument
	 * @return a corresponding KernelArg representing this value
	 */
	public static native KernelArg createValue(float value);

	/**
	 * Create a new FloatArg with the passed values. <br>
	 * This argument will be uploaded, but not be downloaded.
	 * @param floats values for this argument
	 * @return a new FloatArg with the passes values
	 */
	public static FloatArg create(float ...floats) {
		return create(floats, false);
	}

	/**
	 * Create a new FloatArg with the passed values.<br>
	 * This argument will be uploaded. The argument will be downloaded when
	 * <code>download</code> is <code>true</code>.
	 * @param floats values for this argument
	 * @param download set whether the argument should be downloaded
	 * @return a new FloatArg with the passes values
	 */
	public static FloatArg create(float[] floats, boolean download) {
		assert(floats != null && floats.length > 0);

		return createInternal(floats, download);
	}

	private static native FloatArg createInternal(float[] floats, boolean download);

	/**
	 * Create an output-argument, which will be downloaded but not be uploaded. <br>
	 * This argument will be allocating enough memory for <code>length</code> float-elements.
	 * @param length number of elements for the output-argument
	 * @return a new FloatArg
	 */
	public static FloatArg createOutput(int length) {
		return new FloatArg(createOutput(length * SIZE_BYTES));
	}
	
	/**
	 * Create a new HalfArg with the values of this array, which are converted to halfs.
	 * @return a new HalfArg with this data
	 */
	public native HalfArg asHalfArg();

	FloatArg(long handle) {
		super(handle);
	}

	/**
	 * Create a new array containing the data of this array.
	 * @return data of this array
	 */
	public native float[] asFloatArray();
	
	@Override
	protected long getSizeBytes() {
		return SIZE_BYTES;
	}
	
	@Override
	public FloatArg slice(int start, int end) {
		return new FloatArg(slice(start * SIZE_BYTES, end * SIZE_BYTES + SIZE_BYTES));
	}

	@Override
    public String toString(){
        return "FloatArg " + super.toString();
    }
}
