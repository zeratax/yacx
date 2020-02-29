package yacx;

/**
 * Class to instantiate and compile kernel strings.
 */
public class Program extends JNIHandle {

	/**
	 * Create a new Program.
	 * 
	 * @param kernelString string containing the CUDA kernelcode
	 * @param kernelName   name of the kernel
	 * @return a new Program
	 */
	public static Program create(String kernelString, String kernelName) {
		assert (kernelString != null && kernelString.length() > 0);
		assert (kernelName != null && kernelName.length() > 0);

		return createInternal(kernelString, kernelName);
	}

	private static native Program createInternal(String kernelString, String kernelName);

	/**
	 * Create a new Program.
	 * 
	 * @param kernelString string containing the CUDA kernelcode
	 * @param kernelName   name of the kernel
	 * @param headers      headers required by the kernel
	 * @return a new Program
	 */
	public static Program create(String kernelString, String kernelName, Headers headers) {
		assert (kernelString != null && kernelString.length() > 0);
		assert (kernelName != null && kernelName.length() > 0);
		assert (headers != null);

		return createInternal(kernelString, kernelName, headers);
	}

	private static native Program createInternal(String kernelString, String kernelName, Headers headers);

	/**
	 * Instantiate the template parameters if the kernel contains some.
	 * 
	 * @param templateParameter template parameters which should be used for this
	 *                          Program
	 * @return <code>this</code>
	 */
	public Program instantiate(String... templateParameter) {
		assert (templateParameter != null);

		if (templateParameter.length == 0)
			throw new IllegalArgumentException("instantiate template parameters with empty array");

		for (int i = 0; i < templateParameter.length; i++) {
			assert (templateParameter[i] != null);

			instantiateInternal(templateParameter[i]);
		}

		return this;
	}

	/**
	 * Instantiate a template parameter if the kernel contains one or more.
	 * 
	 * @param templateParameter template parameter which should be used for this
	 *                          Program
	 */
	private native void instantiateInternal(String templateParameter);

	/**
	 * Compile this Program using nvrtc.
	 * 
	 * @return compiled Kernel
	 */
	public native Kernel compile();

	/**
	 * Compile this Program using nvrtc.
	 * 
	 * @param options options for the compiler
	 * @return compiled Kernel
	 */
	public Kernel compile(Options options) {
		assert (options != null);

		return compileInternal(options);
	}

	private native Kernel compileInternal(Options options);

	/**
	 * Create a new Program.
	 * 
	 * @param handle Pointer to corresponding C-Program-Object
	 */
	Program(long handle) {
		super(handle);
	}
}
