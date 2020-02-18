package yacx;

/**
 * Class representing an argument for the kernel. <br>
 * Arguments are automatically uploaded and downloaded.
 */
public class KernelArg extends JNIHandle {
	/**
	 * Create a new KernelArg.
	 * 
	 * @param handle Pointer to corresponding C-KernelArg-Object
	 */
	KernelArg(long handle) {
		super(handle);
	}
}
