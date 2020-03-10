package yacx;

import java.util.Arrays;

/**
 * Options for compiling a kernel with nvtrc.
 */
public class Options extends JNIHandle {

	/**
	 * Create a new Options-object with the passing parameters.
	 * 
	 * @param options options for compiling a program
	 * @return new Options-object
	 */
	public static Options createOptions(String... options) {
		Options o = createOptions();

		for (String option : options)
			o.insert(option);

		return o;
	}

	private static native Options createOptions();

	/**
	 * Insert a option for compiling a program.
	 * 
	 * @param option option for Compiler
	 */
	public void insert(String option) {
		assert (option != null && option.length() > 0);

		insertInternal(option);
	}

	private native void insertInternal(String option);

	/**
	 * Insert a option with a specific value for compiling a program.
	 * 
	 * @param name  name of the option
	 * @param value value of the option
	 */
	public void insert(String name, Object value) {
		assert (name != null && name.length() > 0);
		assert (value != null && value.toString().length() > 0);

		insertInternal(name, value.toString());
	}

	private native void insertInternal(String name, String value);

	/**
	 * Returns the number of inserted options.
	 * 
	 * @return number of inserted options
	 */
	public native int getSize();

	/**
	 * Returns the options as Stringarray.
	 * 
	 * @return options as strings
	 */
	public native String[] getOptions();

	/**
	 * Create a new Options-object.
	 * 
	 * @param handle Pointer to corresponding C-Options-Object
	 */
	Options(long handle) {
		super(handle);
	}

	@Override
	public String toString() {
		return "Options " + Arrays.toString(getOptions());
	}
}
