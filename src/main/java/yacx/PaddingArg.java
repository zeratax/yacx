package yacx;

/**
 * Class representing an argument for the kernel. <br>
 * This argument containing an ArrayArg with a specific padding.
 */
public class PaddingArg extends ArrayArg {
	private ArrayArg arg;

	/**
	 * Creates a new PaddingArg with the passed ArrayArg interpreting the array as
	 * matrix. <br>
	 * When uploaded, the PaddingArg create a new array (interpreting as matrix) on
	 * the device with passed values for dimensions using the values from ArrayArg
	 * and <code>paddingValue</code> to fill up remaining values. </br>
	 * if <code>isDownload</code> is true: after kernellaunch the result will be
	 * downloaded to the passed ArrayArg (without the padding). <br>
	 * Only supports arrays with elementsize 1, 2 or 4 bytes.
	 * 
	 * @param arg          ArrayArg
	 * @param rowsArg      number of rows in the matrix <code>arg</code>
	 * @param columnsArg   number of columns in the matrix <code>arg</code>
	 * @param rowsNew      new number of rows for the new matrix on the device
	 * @param columnsNew   new number of columns for the new matrix on the device
	 * @param paddingValue value to fill up remaining for the new matrix
	 * @return new PaddingArg for the passed ArrayArg with specific padding on the
	 *         device
	 */
	public static PaddingArg createMatrixPadding(ArrayArg arg, int rowsArg, int columnsArg, int rowsNew, int columnsNew,
			int paddingValue) {
		assert (columnsArg > 0 && rowsArg > 0);
		assert (columnsNew > 0 && rowsNew > 0);
		assert (columnsNew >= columnsArg && rowsNew >= rowsArg);

		PaddingArg paddingArg = createMatrixPaddingInternal(arg, rowsArg, columnsArg, rowsNew, columnsNew, paddingValue,
				(int) arg.getSizeBytes());
		paddingArg.arg = arg;
		return paddingArg;
	}

	private static native PaddingArg createMatrixPaddingInternal(ArrayArg arg, int rowsArg, int columnsArg, int rowsNew,
			int columnsNew, int paddingValue, int elementSize);

	/**
	 * Create a new PaddingArg.
	 * 
	 * @param handle Pointer to corresponding C-PaddingArg-Object
	 */
	PaddingArg(long handle) {
		super(handle);
	}

	/**
	 * Return original ArrayArg which is padded in this arg. <br>
	 * Use this to access data from this PaddingArg.
	 * 
	 * @return original ArrayArg
	 */
	public ArrayArg getArg() {
		return arg;
	}

	@Override
	protected long getSizeBytes() {
		return arg.getSizeBytes();
	}

	@Override
	public ArrayArg slice(int start, int end) {
		throw new UnsupportedOperationException("slicing a padding arg is not supported");
	}
	
	@Override
	public void dispose() {
		super.dispose();
		arg.dispose();
	}
}
