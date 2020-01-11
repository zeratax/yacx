package yacx;

/**
 * Class representing an half-array-argument for the kernel. <br>
 * The size in bytes of an half-element may depend on your system.
 */
public class HalfArg extends ArrayArg {
	/**
	 * Size in bytes for one element.
	 */
	public final static long SIZE_BYTES = 2;
	
	/**
	 * Convert the float to a half and create a half-value-argument.
	 * @param value value of the argument, which will be convert to half
	 * @return a corresponding KernelArg representing this value
	 */
	public static native KernelArg createValue(float value);

	/**
	 * Create a new HalfArg with the passed values. <br>
	 * This argument will be uploaded, but not be downloaded.
	 * @param floats values for this argument, which will be convert to halfs
	 * @return a new HalfArg with the passes values
	 */
	public static HalfArg create(float ...floats) {
		return create(floats, false);
	}

	/**
	 * Create a new HalfArg with the passed values.<br>
	 * This argument will be uploaded. The argument will be downloaded when
	 * <code>download</code> is <code>true</code>.
	 * @param floats values for this argument, which will be convert to halfs
	 * @param download set whether the argument should be downloaded
	 * @return a new HalfArg with the passes values
	 */
	public static HalfArg create(float[] floats, boolean download) {
		assert(floats != null && floats.length > 0);

		return createInternal(floats, download);
	}

	private static native HalfArg createInternal(float[] floats, boolean download);

	/**
	 * Create an output-argument, which will be downloaded but not be uploaded. <br>
	 * This argument will be allocating enough memory for <code>length</code> half-elements.
	 * @param length number of elements for the output-argument
	 * @return a new HalfArg
	 */
	public static HalfArg createOutput(int length) {
		return new HalfArg(createOutput(length * SIZE_BYTES));
	}

	HalfArg(long handle) {
		super(handle);
	}

	/**
	 * Create a new array containing the data of this array.
	 * @return data of this array, which will be converted to floats
	 */
	public native float[] asFloatArray();
	
	/**
	 * Create a new FloatArg with the values of this array, which are converted to floats.
	 * @return a new FloatArg with this data
	 */
	public native FloatArg asFloatArg();
	
	@Override
	protected long getSizeBytes() {
		return SIZE_BYTES;
	}
	
	@Override
	public HalfArg slice(int start, int end) {
		return new HalfArg(slice(start * SIZE_BYTES, end * SIZE_BYTES + SIZE_BYTES));
	}

	@Override
    public String toString(){
        return "HalfArg " + super.toString();
    }
}
