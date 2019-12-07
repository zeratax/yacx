public class DoubleArg extends KernelArg {
	public static DoubleArg create(double ...doubles) {
		return createInternal(doubles, false);
	}
	
	public static DoubleArg create(double[] doubles, boolean download) {
		assert(doubles != null && doubles.length > 0);
		
		return createInternal(doubles, download);
	}
	
	private static native DoubleArg createInternal(double[] doubles, boolean download);
	
	
	public static DoubleArg createOutput(int size) {
		assert(size > 0);
		
		return createOutputInternal(size);
	}
	
	private static native DoubleArg createOutputInternal(int size);
	
	DoubleArg(long handle) {
		super(handle);
	}
	
	public native double[] asDoubleArray();
	
	@Override
    public String toString(){
        return "DoubleArg " + super.toString();
    }
}
