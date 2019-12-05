public class KernelTime {
    private final float upload;
    private final float download;
    private final float launch;
    private final float sum;

    private KernelTime(float upload, float download, float launch, float sum){
        this.upload = upload;
        this.download = download;
        this.launch = launch;
        this.sum = sum;
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

	public float getSum() {
		return sum;
	}

	@Override
    public String toString(){
        return "Execution-Time: " + sum + " milliseconds (Upload-Time: " + upload +  ", Download-Time: "
            + download + ", Launch-Time: " + launch + ")";
    }
}