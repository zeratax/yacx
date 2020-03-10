package yacx;

/**
 * Class to compile and execute a C-Program.
 */
public class CProgram extends JNIHandle {
	private static final String DEFAULT_COMPILER = "gcc";
	private static final String DEFAULT_OPTIONS = "-Wall -Wextra --pedantic";

	// This header is needed for the primtive datatypes corresponding to the CTYPE
	// of KernelArgs
	// e.g. int32_t for a IntArg
	private static final String HEADER_FOR_DATATYPES = "#include <stdint.h>\n\n";

	/**
	 * Compile a load c cProgram. <br>
	 * The decleration of structs, functions or variables with one of the following
	 * name within passed C-Code is permitted: <br>
	 * op<functionName>, opfn<functionName>, execute<functionName>
	 * 
	 * @param cProgram       string containing the cProgram
	 * @param functionName   name of the cFunction which should be executed
	 * @param parameterTypes types of parameter e.g. int or float* <br>
	 *                       pointer types can be abbreviated by *
	 * @return created and compiled cProgram
	 */
	public static CProgram create(String cProgram, String functionName, String[] parameterTypes) {
		return create(cProgram, functionName, parameterTypes, DEFAULT_COMPILER);
	}

	/**
	 * Compile a load c cProgram. <br>
	 * The decleration of structs, functions or variables with one of the following
	 * name within passed C-Code is permitted: <br>
	 * op<functionName>, opfn<functionName>, execute<functionName>
	 * 
	 * @param cProgram       string containing the cProgram
	 * @param functionName   name of the cFunction which should be executed
	 * @param parameterTypes types of parameter e.g. int or float* <br>
	 *                       pointer types can be abbreviated by *
	 * @param compiler       name of the compiler for compiling the cProgram
	 * @return created and compiled cProgram
	 */
	public static CProgram create(String cProgram, String functionName, String[] parameterTypes, String compiler) {
		return create(cProgram, functionName, parameterTypes, DEFAULT_COMPILER, Options.createOptions(DEFAULT_OPTIONS));
	}

	/**
	 * Compile a load c cProgram. <br>
	 * The decleration of structs, functions or variables with one of the following
	 * name within passed C-Code is permitted: <br>
	 * op<functionName>, opfn<functionName>, execute<functionName>
	 * 
	 * @param cProgram       string containing the cProgram
	 * @param functionName   name of the cFunction which should be executed
	 * @param parameterTypes types of parameter e.g. int or float* <br>
	 *                       pointer types can be abbreviated by *
	 * @param compiler       name of the compiler for compiling the cProgram
	 * @param options        options for the passed compiler
	 * @return created and compiled cProgram
	 */
	public static CProgram create(String cProgram, String functionName, String[] parameterTypes, String compiler,
			Options options) {
		if (cProgram == null)
			throw new NullPointerException();

		assert (!cProgram.equals(""));
		assert (functionName != null && !functionName.equals(""));
		assert (parameterTypes.length > 0);
		assert (compiler != null && !compiler.equals(""));
		assert (options != null);

		return createInternal(HEADER_FOR_DATATYPES + cProgram, functionName, parameterTypes, compiler, options);
	}

	/**
	 * Return the Ctypes of the passed array of KernelArgs.
	 * 
	 * @param args kernelArgs
	 * @return array with corresponding Ctypes
	 */
	public static native String[] getTypes(KernelArg[] args);

	private static native CProgram createInternal(String cProgram, String functionName, String[] parameterTypes,
			String compiler, Options options);

	/**
	 * Execute this cProgram. <br>
	 * The upload and download options of ArrayArgs will be ignored.
	 * 
	 * @param args arguments for function
	 */
	public native void execute(KernelArg... args);

	CProgram(long handle) {
		super(handle);
	}
}
