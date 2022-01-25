package yacx;

import java.io.Serializable;
import java.text.DecimalFormat;

/**
 * Wrapperclass for representing the execution-time of a kernel.
 */
public class KernelTime implements Serializable {
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
	public KernelTime(float upload, float download, float launch, float total, float bandwidthUp,
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
	 * Adds two bandwidths.
	 * 
	 * @param bandwidth1 first bandwidth
	 * @param bandwidth2 second bandwidth
	 * @param time1      duration of first measurement
	 * @param time2      duration of second measurement
	 * @return sum of bandwiths
	 */
	private float addBandwidth(float bandwidth1, float bandwidth2, float time1, float time2) {
		return (bandwidth1 * time1 + bandwidth2 * time2) / (time1 + time2);
	}

	/**
	 * Adds this KernelTime to another KernelTime.
	 * 
	 * @param kernelTime KernelTime, which should be added
	 * @return sum of the kerneltimes
	 */
	public KernelTime addKernelTime(KernelTime kernelTime) {
		return new KernelTime(upload + kernelTime.upload, download + kernelTime.download, launch + kernelTime.launch,
				total + kernelTime.total,
				addBandwidth(bandwidthUp, kernelTime.bandwidthUp, upload, kernelTime.upload),
				addBandwidth(bandwidthDown, kernelTime.bandwidthDown, download, kernelTime.download),
				addBandwidth(bandwidthLaunch, kernelTime.bandwidthLaunch, launch, kernelTime.launch));
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
