package yacx;

/**
 * Class representing an array-argument for the kernel.
 */
public abstract class ArrayArg extends KernelArg {
	/**
	 * Returns the size in bytes of each data-element from the array.
	 * @return size in bytes of each data-element
	 */
	protected abstract long getSizeBytes();
	
	/**
	 * Slices the array. <br>
	 * This creates a new ArrayArg corresponding to a new C-KernelArg-Object, but using the same memory
	 * for the array. <br>
	 * If the content of this ArrayArg changed, the content of the created ArrayArg will be changed too.
	 * In the other direction as well.
	 * @param start index of the first element for the new array
	 * @param end index of the last element for new array
	 * @return a new ArrayArg using the same memory for the array
	 */
	public abstract ArrayArg slice(int start, int end);
	
	/**
	 * Create an output-argument, which will be downloaded but not uploaded.
	 * @param sizeBytes size of the output-argument in bytes
	 * @return a pointer to a new corresponding C-KernelArg-Object
	 */
	protected static long createOutput(long sizeBytes) {
		assert(sizeBytes > 0);

		return createOutputInternal(sizeBytes);
	}
	
	private static native long createOutputInternal(long sizeBytes);
	
	/**
	 * Returns the length of this array-argument.
	 * @return length of this array
	 */
	public int getLength() {
		return (int) (getSize() / getSizeBytes());
	}
	
	/**
	 * Returns the size of the array in bytes.
	 * @return size of the array in bytes
	 */
	private native long getSize();
	
	/**
	 * Returns whether the argument should be downloaded.
	 * @return <code>true</code> if this argument should be downloaded. <code>false</code> otherwise
	 */
	public native boolean isDownload();

	/**
	 * Set whether the argument should be downloaded.
	 * @param download <code>true</code> if this argument should be downloaded. <code>false</code> otherwise
	 */
	public native void setDownload(boolean download);

	/**
	 * Returns whether the argument should be uploaded.
	 * @return <code>true</code> if this argument should be uploaded. <code>false</code> otherwise
	 */
	public native boolean isUpload();

	/**
	 * Set whether the argument should be uploaded.
	 * @param upload <code>true</code> if this argument should be uploaded. <code>false</code> otherwise
	 */
	public native void setUpload(boolean upload);
	
	/**
	 * Create a Slice of this ArrayArg.
	 * @param start index of the first byte for the new array
	 * @param end index of the last byte for new array
	 * @return a pointer to a new C-KernelArg-Object using the same memory for the array
	 */
	protected long slice(long start, long end) {
		assert(start >= 0);
		assert(end >= 0 && end < getLength());
		
		return sliceInternal(start, end);
	}
	
	private native long sliceInternal(long start, long end);
	
	/**
	 * Create a new ArrayArg.
	 * @param handle Pointer to corresponding C-KernelArg-Object for an arary
	 */
	ArrayArg(long handle) {
		super(handle);
	}
}
