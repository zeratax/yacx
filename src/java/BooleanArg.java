public class BooleanArg extends KernelArg {
	public static BooleanArg create(boolean ...booleans) {
		return createInternal(booleans, false);
	}
	
	public static BooleanArg create(boolean[] booleans, boolean download) {
		assert(booleans != null && booleans.length > 0);
		
		return createInternal(booleans, download);
	}
	
	private static native BooleanArg createInternal(boolean[] booleans, boolean download);
	
	
	public static BooleanArg createOutput(int size) {
		assert(size > 0);
		
		return createOutputInternal(size);
	}
	
	private static native BooleanArg createOutputInternal(int size);
	
	BooleanArg(long handle) {
		super(handle);
	}
	
	public native boolean[] asBooleanArray();
	
	@Override
    public String toString(){
        return "BooleanArg " + super.toString();
    }
}
