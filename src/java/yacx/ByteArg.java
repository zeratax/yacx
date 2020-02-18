package yacx;

/**
 * Class representing an byte-array-argument for the kernel. <br>
 * Note: The size in bytes for one byte-element is fixed and may differ from the
 * size of the corresponding data-type in CUDA, which is depending on your
 * system. So make sure the size of the corresponding data-type is matching
 * {@link #SIZE_BYTES} to avoid unexpected segmentation faults.
 */
public class ByteArg extends ArrayArg {
	/**
	 * Size in bytes for one element.
	 */
	public final static long SIZE_BYTES = 1;

	/**
	 * Create a byte-value-argument.
	 * 
	 * @param value value of the argument
	 * @return a corresponding KernelArg representing this value
	 */
	public static native KernelArg createValue(byte value);

	/**
	 * Create a new ByteArg with the passed values. <br>
	 * This argument will be uploaded, but not be downloaded.
	 * 
	 * @param bytes values for this argument
	 * @return a new ByteArg with the passes values
	 */
	public static ByteArg create(byte... bytes) {
		return create(bytes, false);
	}

	/**
	 * Create a new ByteArg with the passed values.<br>
	 * This argument will be uploaded. The argument will be downloaded when
	 * <code>download</code> is <code>true</code>.
	 * 
	 * @param bytes    values for this argument
	 * @param download set whether the argument should be downloaded
	 * @return a new ByteArg with the passes values
	 */
	public static ByteArg create(byte[] bytes, boolean download) {
		assert (bytes != null && bytes.length > 0);

		return createInternal(bytes, download);
	}

	private static native ByteArg createInternal(byte[] bytes, boolean download);

	/**
	 * Create an output-argument, which will be downloaded but not be uploaded. <br>
	 * This argument will be allocating enough memory for <code>length</code>
	 * byte-elements.
	 * 
	 * @param length number of elements for the output-argument
	 * @return a new ByteArg
	 */
	public static ByteArg createOutput(int length) {
		return new ByteArg(createOutput(length * SIZE_BYTES));
	}

	ByteArg(long handle) {
		super(handle);
	}

	/**
	 * Create a new array containing the data of this array.
	 * 
	 * @return data of this array
	 */
	public native byte[] asByteArray();

	@Override
	protected long getSizeBytes() {
		return SIZE_BYTES;
	}

	@Override
	public ByteArg slice(int start, int end) {
		return new ByteArg(slice(start * SIZE_BYTES, end * SIZE_BYTES + SIZE_BYTES));
	}

	@Override
	public String toString() {
		return "ByteArg " + super.toString();
	}
}
