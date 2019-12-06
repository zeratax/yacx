public class ShortArg extends KernelArg {
	public static ShortArg create(short ...shorts) {
		return createInternal(shorts, false);
	}
	
	public static ShortArg create(short[] shorts, boolean download) {
		assert(shorts != null && shorts.length > 0);
		
		return createInternal(shorts, download);
	}
	
	private static native ShortArg createInternal(short[] shorts, boolean download);
	
	
	public static ShortArg createOutput(int size) {
		assert(size > 0);
		
		return createOutputInternal(size);
	}
	
	private static native ShortArg createOutputInternal(int size);
	
	ShortArg(long handle) {
		super(handle);
	}
	
	public native short[] asShortArray();
	
	@Override
    public String toString(){
        return "ShortArg " + super.toString();
    }
}
