package yacx;

import java.io.IOException;
import java.text.DecimalFormat;
import java.util.Arrays;

public class Executor {
    public static void loadLibary() {
        System.loadLibrary("yacx-jni");
    }

    public static KernelTime launch(String kernelName, int grid, int block, KernelArg ...args) throws IOException {
    	return launch(Utils.loadFile(kernelName + ".cu"), kernelName, grid, block, args);
    }

    public static KernelTime launch(String kernelName, Options options, int grid, int block, KernelArg ...args) throws IOException {
    	return launch(Utils.loadFile(kernelName + ".cu"), kernelName, options, grid, block, args);
    }

    public static KernelTime launch(String kernelName, Options options, String deviceName, int grid, int block, KernelArg ...args) throws IOException {
        return launch(Utils.loadFile(kernelName + ".cu"), kernelName, options, deviceName, grid, block, args);
    }

    public static KernelTime launch(String kernelString, String kernelName, int grid, int block, KernelArg ...args) {
    	return Program.create(kernelString, kernelName).compile().configure(grid, block).launch(args);
    }

    public static KernelTime launch(String kernelString, String kernelName, Options options, int grid, int block, KernelArg ...args) {
    	return Program.create(kernelString, kernelName).compile(options).configure(grid, block).launch(args);
    }

    public static KernelTime launch(String kernelString, String kernelName, Options options, String deviceName, int grid, int block, KernelArg ...args) {
    	return Program.create(kernelString, kernelName).compile(options).configure(grid, block).launch(Device.createDevice(deviceName), args);
    }

    public static KernelTime launch(String kernelName, int grid0, int grid1, int grid2, int block0, int block1, int block2, KernelArg ...args) throws IOException {
    	return launch(Utils.loadFile(kernelName + ".cu"), kernelName, grid0, grid1, grid2, block0, block1, block2, args);
    }

    public static KernelTime launch(String kernelName, Options options, int grid0, int grid1, int grid2, int block0, int block1, int block2, KernelArg ...args) throws IOException {
    	return launch(Utils.loadFile(kernelName + ".cu"), kernelName, options, grid0, grid1, grid2, block0, block1, block2, args);
    }

    public static KernelTime launch(String kernelName, Options options, String deviceName, int grid0, int grid1, int grid2, int block0, int block1, int block2, KernelArg ...args) throws IOException {
      	return launch(Utils.loadFile(kernelName + ".cu"), kernelName, options, deviceName, grid0, grid1, grid2, block0, block1, block2, args);
    }

    public static KernelTime launch(String kernelString, String kernelName, int grid0, int grid1, int grid2, int block0, int block1, int block2, KernelArg ...args) {
    	return Program.create(kernelString, kernelName).compile().configure(grid0, grid1, grid2, block0, block1, block2).launch(args);
    }

    public static KernelTime launch(String kernelString, String kernelName, Options options, int grid0, int grid1, int grid2, int block0, int block1, int block2, KernelArg ...args) {
    	return Program.create(kernelString, kernelName).compile(options).configure(grid0, grid1, grid2, block0, block1, block2).launch(args);
    }

    public static KernelTime launch(String kernelString, String kernelName, Options options, String deviceName, int grid0, int grid1, int grid2, int block0, int block1, int block2, KernelArg ...args) {
    	return Program.create(kernelString, kernelName).compile(options).configure(grid0, grid1, grid2, block0, block1, block2).launch(Device.createDevice(deviceName), args);
    }


    public static BenchmarkResult benchmark(String kernelString, String kernelName, Options options, int numberExecutions, KernelArgCreator creator, int ...dataSizesBytes) {
    	return benchmark(kernelString, kernelName, options, Device.createDevice(), numberExecutions, creator, dataSizesBytes);
    }

    public static BenchmarkResult benchmark(String kernelString, String kernelName, Options options, Device device, int numberExecutions, KernelArgCreator creator, int ...dataSizesBytes) {
    	if (dataSizesBytes == null)
    		throw new NullPointerException();
    	if (dataSizesBytes.length == 0)
    		throw new IllegalArgumentException("not data sizes specificated");
    	if (numberExecutions <= 0)
    		throw new IllegalArgumentException("illegal number of executions: " + numberExecutions);

    	//Absolute time Measurement
    	long t0 = System.currentTimeMillis();

    	//Create and compile Kernel
    	Kernel kernel = Program.create(kernelString, kernelName)
    					.compile(options);

    	//Array for result
    	KernelTime[][] result = new KernelTime[dataSizesBytes.length][numberExecutions];

    	//Start run for every dataSize
    	for (int i = 0; i < dataSizesBytes.length; i++) {
    		int dataSize = dataSizesBytes[i];

    		if (dataSize <= 0)
    			throw new IllegalArgumentException();

    		//Configure Kernel
    		int dataLength = creator.getDataLength(dataSize);
    		kernel.configure(creator.getGrid0(dataLength), creator.getGrid1(dataLength), creator.getGrid2(dataLength),
    							creator.getBlock0(dataLength), creator.getBlock1(dataLength), creator.getBlock2(dataLength));

    		//Execute Kernel numberExecutions times
    		result[i] = benchmark(kernel, device, creator.createArgs(dataLength), numberExecutions);
    	}

    	//Absolute time Measurement
    	long dt = System.currentTimeMillis()-t0;

    	return new BenchmarkResult(numberExecutions, dataSizesBytes, result, kernelName, dt);
    }

    private static native KernelTime[] benchmark(Kernel kernel, Device device, KernelArg[] args, int numberExecutions);

    public static abstract class KernelArgCreator {
    	public abstract int getDataLength(int dataSizeBytes);
    	public abstract KernelArg[] createArgs(int dataLength);
    	public abstract int getGrid0(int dataLength);
    	public int getGrid1(int dataLength) {return 1;}
    	public int getGrid2(int dataLength) {return 1;}
    	public abstract int getBlock0(int dataLength);
    	public int getBlock1(int dataLength) {return 1;}
    	public int getBlock2(int dataLength) {return 1;}
    }

    public static class BenchmarkResult {
    	private final int executions;
    	private final int[] dataSizes;
    	private final KernelTime[][] result;
    	private final KernelTime[] average;
    	private final String kernelName;
    	private final long testDuration;

    	protected BenchmarkResult(int executions, int[] dataSizes, KernelTime[][] result, String kernelName, long testDuration) {
    		this.executions = executions;
    		this.dataSizes = dataSizes;
    		Arrays.parallelSort(dataSizes);
    		this.result = result;
    		this.kernelName = kernelName;
    		this.testDuration = testDuration;

    		//Compute Average
    		average = new KernelTime[result.length];
    		for (int i = 0; i < dataSizes.length; i++) {
    			double upload = 0;
    			double download = 0;
    			double launch = 0;
    			double sum = 0;

    			for (int j = 0; j < executions; j++) {
    				upload += result[i][j].getUpload();
    				download += result[i][j].getDownload();
    				launch += result[i][j].getLaunch();
    				sum += result[i][j].getSum();
    			}

    			average[i] = new KernelTime((float) (upload/executions), (float) (download/executions),
    					(float) (launch/executions), (float) (sum/executions));
    		}
    	}

		public int getExecutions() {
			return executions;
		}

		public int[] getDataSizes() {
			return dataSizes;
		}

		public KernelTime[][] getResult() {
			return result;
		}

		public KernelTime[] getAverage() {
			return average;
		}

		public String getKernelName() {
			return kernelName;
		}

		@Override
		public String toString() {
			StringBuffer buffer = new StringBuffer(200);
			buffer.append("\nBenchmark " + kernelName + "-Kernel\n");

			buffer.append("  Datasize  Result (Average)\n");

			for (int i = 0; i < dataSizes.length; i++) {
				String dataSize = "" + humanReadableByteCountBin(dataSizes[i]);
				while (dataSize.length() < 10)
					dataSize = " " + dataSize;

				String result = average[i].toString();

				buffer.append(dataSize);
				buffer.append(": ");
				buffer.append(result);
				buffer.append("\n");
			}

			DecimalFormat df = new DecimalFormat();
			String time = KernelTime.humanReadableMilliseconds(df, testDuration);
			df.setMaximumFractionDigits(1);
			String[] s = time.split(" ");

			buffer.append("\nBenchmark-Duration: " + df.format(Double.parseDouble(s[0])) + " " + s[2] + "\n");

			return buffer.toString();
		}

		static String humanReadableByteCountBin(long bytes) {
		    return bytes < 1024L ? bytes + " B"
		            : bytes <= 0xfffccccccccccccL >> 40 ? String.format("%.1f KiB", bytes / 0x1p10)
		            : bytes <= 0xfffccccccccccccL >> 30 ? String.format("%.1f MiB", bytes / 0x1p20)
		            : String.format("%.1f GiB", bytes / 0x1p30);
		}
    }
}
