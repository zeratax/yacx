package yacx;

/**
 * Exception class for Errors occur during execution of a kernel. <br>
 * Especially errors while compiling kernel with nvrtc or while using CUDA driver api create
 * this Exception. <br>
 * Errors like segmentation faults (e.g. passing invalid arguments to CUDA kernel) are
 * <strong>not</strong> caught and will crash the execution of the running program.
 */
public class ExecutorFailureException extends RuntimeException {
	/**
	 * Create a new ExecutorFailureException.
	 * @param message useful message describing the error
	 */
	public ExecutorFailureException(String message){
        super(message);
        System.err.println("Runtime error in the executor");
    }
}
