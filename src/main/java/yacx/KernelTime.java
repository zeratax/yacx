package yacx;

import java.text.DecimalFormat;

/**
 * Wrapperclass for representing the execution-time of a kernel.
 */
public class KernelTime {
	private final float upload;
	private final float download;
	private final float launch;
	private final float total;
	private final float bandwidthUp;
	private final float bandwidthDown;
	private final float bandwidthLaunch;

	/**
	 * Create a new KernelTime.
	 * 
	 * @param upload          time to upload the arguments in milliseconds
	 * @param download        time to download the arguments in milliseconds
	 * @param launch          time to launch the kernel in milliseconds
	 * @param total           total time of execution in milliseconds
	 * 
	 * @param bandwidthUp     bandwidth of uploading arguments
	 * @param bandwidthDown   bandwidth of download arguments
	 * @param bandwidthLaunch bandwidth of launching kernel inclusive uploading and
	 *                        downloading arguments
	 */
	protected KernelTime(float upload, float download, float launch, float total, float bandwidthUp,
			float bandwidthDown, float bandwidthLaunch) {
		this.upload = upload;
		this.download = download;
		this.launch = launch;
		this.total = total;

		this.bandwidthUp = bandwidthUp;
		this.bandwidthDown = bandwidthDown;
		this.bandwidthLaunch = bandwidthLaunch;
	}

	/**
	 * Returns the time to upload the arguments in milliseconds.
	 * 
	 * @return time to upload the arguments in milliseconds
	 */
	public float getUpload() {
		return upload;
	}

	/**
	 * Returns the time to download the arguments in milliseconds.
	 * 
	 * @return time to download the arguments in milliseconds
	 */
	public float getDownload() {
		return download;
	}

	/**
	 * Returns the time to launch the kernel in milliseconds.
	 * 
	 * @return time to launch the kernel in milliseconds
	 */
	public float getLaunch() {
		return launch;
	}

	/**
	 * Returns the total time of executions (upload arguments, launch kernel,
	 * download result) in milliseconds.
	 * 
	 * @return total time of execution in milliseconds
	 */
	public float getTotal() {
		return total;
	}

	/**
	 * Returns the effective bandwidth of uploading arguments.
	 * 
	 * @return bandwidth in GB per second
	 */
	public float effectiveBandwithUpload() {
		return bandwidthUp;
	}

	/**
	 * Returns the effective bandwidth of downloading arguments.
	 * 
	 * @return bandwidth in GB per second
	 */
	public float effectiveBandwithDownload() {
		return bandwidthDown;
	}

	/**
	 * Returns the effective bandwidth of launching kernel.
	 * 
	 * @return bandwidth in GB per second
	 */
	public float effectiveBandwithLaunch() {
		return bandwidthLaunch;
	}

	@Override
	public String toString() {
		DecimalFormat df = new DecimalFormat();

		return "execution-time: " + humanReadableMilliseconds(df, launch) + " (total time: "
				+ humanReadableMilliseconds(df, total) + ", upload-time: " + humanReadableMilliseconds(df, upload)
				+ ", download-time: " + humanReadableMilliseconds(df, download) + ", " + effectiveBandwithLaunch()
				+ " GB/s)";
	}

	static String humanReadableMilliseconds(DecimalFormat df, double time) {
		String unit = "ms";
		if (time > 1000) {
			time /= 1000;
			unit = "s";

			if (time > 100) {
				time /= 60;
				unit = "m";
			}
			if (time > 100) {
				time /= 60;
				unit = "h";
			}
		}

		for (int i = 10, j = 5; true; i *= 10, j--) {
			if (time < i) {
				df.setMinimumFractionDigits(j);
				df.setMaximumFractionDigits(j);

				return df.format(time) + " " + unit;
			}
		}
	}
}
