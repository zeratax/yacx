import java.io.IOException;
import java.text.DecimalFormat;
import java.util.Arrays;

public class Executor {
    public static void loadLibary() {
        System.loadLibrary("cudaexecutor-jni");
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
    
    
    public static BenchmarkResult benchmark(String kernelString, String kernelName, Options options, int numberExecutions, KernelArgCreator creator, int ...dataSizes) {
    	return benchmark(kernelString, kernelName, options, Device.createDevice().getName(), numberExecutions, creator, dataSizes);
    }
    
    public static BenchmarkResult benchmark(String kernelString, String kernelName, Options options, String deviceName, int numberExecutions, KernelArgCreator creator, int ...dataSizes) {
    	if (dataSizes == null)
    		throw new NullPointerException();
    	if (dataSizes.length == 0)
    		throw new IllegalArgumentException("not data sizes specificated");
    	
    	Kernel kernel = Program.create(kernelString, kernelName)
    					.compile(options);
    	Device device = Device.createDevice(deviceName);
    	
    	KernelTime[][] result = new KernelTime[dataSizes.length][numberExecutions];
    	
    	for (int i = 0; i < dataSizes.length; i++) {
    		int dataSize = dataSizes[i];
    		
    		kernel.configure(creator.getGrid0(dataSize), creator.getGrid1(dataSize), creator.getGrid2(dataSize),
    							creator.getBlock0(dataSize), creator.getBlock1(dataSize), creator.getBlock2(dataSize));
    		result[i] = benchmark(kernel, device, creator.createArgs(dataSize), numberExecutions);
    	}
    	
    	return new BenchmarkResult(numberExecutions, dataSizes, result, kernelName);
    }
    
    private static native KernelTime[] benchmark(Kernel kernel, Device device, KernelArg[] args, int numberExecutions);
    
    public static abstract class KernelArgCreator {
    	public abstract KernelArg[] createArgs(int dataSize);
    	public abstract int getGrid0(int dataSize);
    	public int getGrid1(int dataSize) {return 1;}
    	public int getGrid2(int dataSize) {return 1;}
    	public abstract int getBlock0(int dataSize);
    	public int getBlock1(int dataSize) {return 1;}
    	public int getBlock2(int dataSize) {return 1;}
    }
    
    public static class BenchmarkResult {
    	private final int executions;
    	private final int[] dataSizes;
    	private final KernelTime[][] result;
    	private final KernelTime[] average;
    	private final String kernelName;
    	
    	protected BenchmarkResult(int executions, int[] dataSizes, KernelTime[][] result, String kernelName) {
    		this.executions = executions;
    		this.dataSizes = dataSizes;
    		Arrays.parallelSort(dataSizes);
    		this.result = result;
    		this.kernelName = kernelName;
    		
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
			buffer.append("Benchmark " + kernelName + "-Kernel\n");
				
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
			
			return buffer.toString();
		}
			
		private String humanReadableByteCountBin(long bytes) {
		    return bytes < 1024L ? bytes + " B"
		            : bytes <= 0xfffccccccccccccL >> 40 ? String.format("%.1f KiB", bytes / 0x1p10)
		            : bytes <= 0xfffccccccccccccL >> 30 ? String.format("%.1f MiB", bytes / 0x1p20)
		            : String.format("%.1f GiB", bytes / 0x1p30);
		}
    }
}







