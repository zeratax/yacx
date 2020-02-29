package yacx;

/**
 * Class for handle C-Objects.
 */
public abstract class JNIHandle {
	/**
	 * Create a new JNIHandle.
	 * 
	 * @param handle Pointer to C-Object
	 */
	JNIHandle(long handle) {
		nativeHandle = handle;
	}

	/**
	 * Delete corresponding C-Object.
	 */
	public native void dispose();

	private final long nativeHandle; // Pointer to C-Object

	@Override
	public boolean equals(Object o) {
		return o != null && o instanceof JNIHandle && nativeHandle == ((JNIHandle) o).nativeHandle;
	}

	@Override
	public String toString() {
		return String.format("0x%08X", nativeHandle);
	}
}
