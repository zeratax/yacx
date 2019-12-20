import java.text.DecimalFormat;

public class KernelTime {
    private final float upload;
    private final float download;
    private final float launch;
    private final float total;

    protected KernelTime(float upload, float download, float launch, float total){
        this.upload = upload;
        this.download = download;
        this.launch = launch;
        this.total = total;
    }

    public float getUpload() {
		return upload;
	}

	public float getDownload() {
		return download;
	}

	public float getLaunch() {
		return launch;
	}

	public float gettotal() {
		return total;
	}

	@Override
    public String toString(){
		DecimalFormat df = new DecimalFormat();

        return "execution-time: " + humanReadableMilliseconds(df, launch) + " (total time: " +
        			humanReadableMilliseconds(df, total) + ", upload-time: " + humanReadableMilliseconds(df, upload) +
        			", download-time: " + humanReadableMilliseconds(df, download) + ")";
    }

	static String humanReadableMilliseconds(DecimalFormat df, double time) {
		String unit = "ms";
		if (time > 1000) {
			time /= 1000;
			unit = " s";

			if (time > 100) {
				time /= 60;
				unit = " m";
			}
			if (time > 100) {
				time /= 60;
				unit = " h";
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
