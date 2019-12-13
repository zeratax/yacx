public class ByteArg extends KernelArg {
	public final static int SIZE_BYTES = 1;
	
	public static ByteArg create(byte ...bytes) {
		return createInternal(bytes, false);
	}
	
	public static ByteArg create(byte[] bytes, boolean download) {
		assert(bytes != null && bytes.length > 0);
		
		return createInternal(bytes, download);
	}
	
	private static native ByteArg createInternal(byte[] bytes, boolean download);
	
	
	public static ByteArg createOutput(int size) {
		assert(size > 0);
		
		return createOutputInternal(size);
	}
	
	private static native ByteArg createOutputInternal(int length);
	
	ByteArg(long handle) {
		super(handle);
	}
	
	public native byte[] asByteArray();
}
