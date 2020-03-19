package yacx;

public class Logger {
	enum LogLevel {
		/** don't log at all */
		NONE,
		/** an ERROR which should not be ignored */
		ERROR,
		/** a WARNING which might be ignored */
		WARNING,
		/** a INFO which can be ignored */
		INFO,
		/** verbose INFO which can be ignored */
		DEBUG,
		/** verbose DEBUG which can be ignored */
		DEBUG1
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
