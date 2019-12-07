import java.io.IOException;

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
    	return Program.create(kernelString, kernelName).compile(options).configure(grid, block).launch(args, Device.createDevice(deviceName));
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
    	return Program.create(kernelString, kernelName).compile(options).configure(grid0, grid1, grid2, block0, block1, block2).launch(args, Device.createDevice(deviceName));
    }
}