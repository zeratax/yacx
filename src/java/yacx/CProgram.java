package yacx;

/**
 * 
 */
public class CProgram extends JNIHandle {
	private static final String DEFAULT_COMPILER = "gcc";
	private static final String DEFAULT_OPTIONS = "-Wall -Wextra";

	/**
	 * Compile a load c cProgram.
	 * 
	 * @param cProgram        string containing the cProgram
	 * @param functionName    name of the cFunction which should be executed
	 * @param numberParameter number of parameters for the cFunction
	 * @return created and compiled cProgram
	 */
	public static CProgram create(String cProgram, String functionName, int numberParameter) {
		return create(cProgram, functionName, numberParameter, DEFAULT_COMPILER);
	}

	/**
	 * Compile a load c cProgram.
	 * 
	 * @param cProgram        string containing the cProgram
	 * @param functionName    name of the cFunction which should be executed
	 * @param numberParameter number of parameters for the cFunction
	 * @param compiler        name of the compiler for compiling the cProgram
	 * @return created and compiled cProgram
	 */
	public static CProgram create(String cProgram, String functionName, int numberParameter, String compiler) {
		return create(cProgram, functionName, numberParameter, DEFAULT_COMPILER,
				Options.createOptions(DEFAULT_OPTIONS));
	}

	/**
	 * Compile a load c cProgram.
	 * 
	 * @param cProgram        string containing the cProgram
	 * @param functionName    name of the cFunction which should be executed
	 * @param numberParameter number of parameters for the cFunction
	 * @param compiler        name of the compiler for compiling the cProgram
	 * @param options         options for the passed compiler
	 * @return created and compiled cProgram
	 */
	public static CProgram create(String cProgram, String functionName, int numberParameter, String compiler,
			Options options) {
		assert (cProgram != null && !cProgram.equals(""));
		assert (functionName != null && !functionName.equals(""));
		assert (numberParameter > 0);
		assert (compiler != null && !compiler.equals(""));
		assert (options != null);

		return createInternal(cProgram, functionName, numberParameter, compiler, options);
	}

	private static native CProgram createInternal(String cProgram, String functionName, int numberParameter,
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
