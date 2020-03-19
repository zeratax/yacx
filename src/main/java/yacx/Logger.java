package yacx;

public class Logger {
	enum LogLevel {
		NONE,
		/** < don't log at all */
		ERROR,
		/** < an ERROR which should not be ignored */
		WARNING,
		/** < a WARNING which might be ignored */
		INFO,
		/** < a INFO which can be ignored */
		DEBUG,
		/** < verbose INFO which can be ignored */
		DEBUG1 /** < verbose DEBUG which can be ignored */
	}

	/**
	 * Set the LogLevel for the Logger.<br>
	 * Default LogLevel is <code>LogLevel.ERROR</code>.
	 * 
	 * @param level new LogLevel for Logger
	 */
	public static native void setLogLevel(LogLevel level);

	/**
	 * Set/Unset standard output as a logging output.<br>
	 * Default value is <code>true</code>.
	 * 
	 * @param flag <code>true</code> if the log should be printed in standard
	 *             output, <code>false</code> otherwise
	 */
	public static native void setCout(boolean flag);

	/**
	 * Set/Unset standard error output as a logging output.<br>
	 * Default value is <code>true</code>.
	 * 
	 * @param flag <code>true</code> if the log should be printed in standard error
	 *             output, <code>false</code> otherwise
	 */
	public static native void setCerr(boolean flag);

	/**
	 * Set a logfile for the Logger.<br>
	 * Default there is no logfile.
	 * 
	 * @param filename filename of new logfile
	 */
	public static native void setLogfile(String filename);
}
