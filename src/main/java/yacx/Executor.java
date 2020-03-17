package yacx;

import java.io.IOException;
import java.text.DecimalFormat;
import java.util.Arrays;

/**
 * Class for initialization (load native library) and easily execute CUDA
 * kernels. <br>
 * Before using other classes in this package it is necessary to load the native
 * libary.
 */
public class Executor {
	/**
	 * Loads the native library.
	 */
	public static void loadLibary() {
		System.loadLibrary("yacx-jni");
	}

	/**
	 * Launch a CFunction.
	 * 
	 * @param cProgram     string containing the cProgram
	 * @param functionName name of the function
	 * @param args         arguments for function
	 * @return execution time in milliseconds (wallclock time)
	 */
	public static long executeC(String cProgram, String functionName, KernelArg... args) {
		CProgram cProg = CProgram.create(cProgram, functionName, CProgram.getTypes(args));

		long t0 = System.currentTimeMillis();
		cProg.execute(args);
		return System.currentTimeMillis() - t0;
	}

	/**
	 * Launch a CFunction.
	 * 
	 * @param cProgram     string containing the cProgram
	 * @param functionName name of the function
	 * @param compiler     name of the compiler for compiling the cProgram
	 * @param args         arguments for function
	 * @return execution time in milliseconds (wallclock time)
	 */
	public static long executeC(String cProgram, String functionName, String compiler, KernelArg... args) {
		CProgram cProg = CProgram.create(cProgram, functionName, CProgram.getTypes(args), compiler);

		long t0 = System.currentTimeMillis();
		cProg.execute(args);
		return System.currentTimeMillis() - t0;
	}

	/**
	 * Launch a CFunction.
	 * 
	 * @param cProgram     string containing the cProgram
	 * @param functionName name of the function
	 * @param compiler     name of the compiler for compiling the cProgram
	 * @param options      options for the passed compiler
	 * @param args         arguments for function
	 * @return execution time in milliseconds (wallclock time)
	 */
	public static long executeC(String cProgram, String functionName, String compiler, Options options,
			KernelArg... args) {
		CProgram cProg = CProgram.create(cProgram, functionName, CProgram.getTypes(args), compiler, options);

		long t0 = System.currentTimeMillis();
		cProg.execute(args);
		return System.currentTimeMillis() - t0;
	}

	/**
	 * Launch a CUDA kernel loading the kernel string from a file in directory
	 * "kernels" with kernelname.cu as filename.
	 * 
	 * @param kernelName name of the kernel
	 * @param grid       number of grids for kernellaunch
	 * @param block      number of blocks for kernellaunch
	 * @param args       KernelArgs
	 * @return KernelTime for the Execution of this Kernel
	 */
	public static KernelTime launch(String kernelName, int grid, int block, KernelArg... args) throws IOException {
		return launch(Utils.loadFile("kernels/" + kernelName + ".cu"), kernelName, grid, block, args);
	}

	/**
	 * Launch a CUDA kernel loading the kernel string from a file in directory
	 * "kernels" with kernelname.cu as filename.
	 * 
	 * @param kernelName name of the kernel
	 * @param options    options for the nvtrc compiler
	 * @param grid       number of grids for kernellaunch
	 * @param block      number of blocks for kernellaunch
	 * @param args       KernelArgs
	 * @return KernelTime for the Execution of this Kernel
	 */
	public static KernelTime launch(String kernelName, Options options, int grid, int block, KernelArg... args)
			throws IOException {
		return launch(Utils.loadFile("kernels/" + kernelName + ".cu"), kernelName, options, grid, block, args);
	}

	/**
	 * Launch a CUDA kernel loading the kernel string from a file in directory
	 * "kernels" with kernelname.cu as filename.
	 * 
	 * @param kernelName name of the kernel
	 * @param options    options for the nvtrc compiler
	 * @param deviceName name of the device on which the kernel should be launched
	 * @param grid       number of grids for kernellaunch
	 * @param block      number of blocks for kernellaunch
	 * @param args       KernelArgs
	 * @return KernelTime for the Execution of this Kernel
	 */
	public static KernelTime launch(String kernelName, Options options, String deviceName, int grid, int block,
			KernelArg... args) throws IOException {
		return launch(Utils.loadFile("kernels/" + kernelName + ".cu"), kernelName, options, deviceName, grid, block,
				args);
	}

	/**
	 * Launch a CUDA kernel.
	 * 
	 * @param kernelString string containing the CUDA kernelcode
	 * @param kernelName   name of the kernel
	 * @param grid         number of grids for kernellaunch
	 * @param block        number of blocks for kernellaunch
	 * @param args         KernelArgs
	 * @return KernelTime for the Execution of this Kernel
	 */
	public static KernelTime launch(String kernelString, String kernelName, int grid, int block, KernelArg... args) {
		return Program.create(kernelString, kernelName).compile().configure(grid, block).launch(args);
	}

	/**
	 * Launch a CUDA kernel.
	 * 
	 * @param kernelString string containing the CUDA kernelcode
	 * @param kernelName   name of the kernel
	 * @param options      options for the nvtrc compiler
	 * @param grid         number of grids for kernellaunch
	 * @param block        number of blocks for kernellaunch
	 * @param args         KernelArgs
	 * @return KernelTime for the Execution of this Kernel
	 */
	public static KernelTime launch(String kernelString, String kernelName, Options options, int grid, int block,
			KernelArg... args) {
		return Program.create(kernelString, kernelName).compile(options).configure(grid, block).launch(args);
	}

	/**
	 * Launch a CUDA kernel.
	 * 
	 * @param kernelString string containing the CUDA kernelcode
	 * @param kernelName   name of the kernel
	 * @param options      options for the nvtrc compiler
	 * @param deviceName   name of the device on which the kernel should be launched
	 * @param grid         number of grids for kernellaunch
	 * @param block        number of blocks for kernellaunch
	 * @param args         KernelArgs
	 * @return KernelTime for the Execution of this Kernel
	 */
	public static KernelTime launch(String kernelString, String kernelName, Options options, String deviceName,
			int grid, int block, KernelArg... args) {
		return Program.create(kernelString, kernelName).compile(options).configure(grid, block).launch(deviceName,
				args);
	}

	/**
	 * Launch a CUDA kernel with template parameters.
	 * 
	 * @param kernelString      string containing the CUDA kernelcode
	 * @param kernelName        name of the kernel
	 * @param options           options for the nvtrc compiler
	 * @param deviceName        name of the device on which the kernel should be
	 *                          launched
	 * @param templateParameter array of templateParameters which can not be empty
	 * @param grid              number of grids for kernellaunch
	 * @param block             number of blocks for kernellaunch
	 * @param args              KernelArgs
	 * @return KernelTime for the Execution of this Kernel
	 */
	public static KernelTime launch(String kernelString, String kernelName, Options options, String deviceName,
			String[] templateParameter, int grid, int block, KernelArg... args) {
		return Program.create(kernelString, kernelName).instantiate(templateParameter).compile(options)
				.configure(grid, block).launch(deviceName, args);
	}

	/**
	 * Launch a CUDA kernel.
	 * 
	 * @param kernelString string containing the CUDA kernelcode
	 * @param kernelName   name of the kernel
	 * @param grid0        number of grids for kernellaunch in first dimension
	 * @param grid1        number of grids for kernellaunch in second dimension
	 * @param grid2        number of grids for kernellaunch in third dimension
	 * @param block0       number of blocks for kernellaunch in first dimension
	 * @param block1       number of blocks for kernellaunch in second dimension
	 * @param block2       number of blocks for kernellaunch in third dimension
	 * @param args         KernelArgs
	 * @return KernelTime for the Execution of this Kernel
	 */
	public static KernelTime launch(String kernelString, String kernelName, int grid0, int grid1, int grid2, int block0,
			int block1, int block2, KernelArg... args) {
		return Program.create(kernelString, kernelName).compile().configure(grid0, grid1, grid2, block0, block1, block2)
				.launch(args);
	}

	/**
	 * Launch a CUDA kernel.
	 * 
	 * @param kernelString string containing the CUDA kernelcode
	 * @param kernelName   name of the kernel
	 * @param options      options for the nvtrc compiler
	 * @param grid0        number of grids for kernellaunch in first dimension
	 * @param grid1        number of grids for kernellaunch in second dimension
	 * @param grid2        number of grids for kernellaunch in third dimension
	 * @param block0       number of blocks for kernellaunch in first dimension
	 * @param block1       number of blocks for kernellaunch in second dimension
	 * @param block2       number of blocks for kernellaunch in third dimension
	 * @param args         KernelArgs
	 * @return KernelTime for the Execution of this Kernel
	 */
	public static KernelTime launch(String kernelString, String kernelName, Options options, int grid0, int grid1,
			int grid2, int block0, int block1, int block2, KernelArg... args) {
		return Program.create(kernelString, kernelName).compile(options)
				.configure(grid0, grid1, grid2, block0, block1, block2).launch(args);
	}

	/**
	 * Launch a CUDA kernel.
	 * 
	 * @param kernelString string containing the CUDA kernelcode
	 * @param kernelName   name of the kernel
	 * @param options      options for the nvtrc compiler
	 * @param deviceName   name of the device on which the kernel should be launched
	 * @param grid0        number of grids for kernellaunch in first dimension
	 * @param grid1        number of grids for kernellaunch in second dimension
	 * @param grid2        number of grids for kernellaunch in third dimension
	 * @param block0       number of blocks for kernellaunch in first dimension
	 * @param block1       number of blocks for kernellaunch in second dimension
	 * @param block2       number of blocks for kernellaunch in third dimension
	 * @param args         KernelArgs
	 * @return KernelTime for the Execution of this Kernel
	 */
	public static KernelTime launch(String kernelString, String kernelName, Options options, String deviceName,
			int grid0, int grid1, int grid2, int block0, int block1, int block2, KernelArg... args) {
		return Program.create(kernelString, kernelName).compile(options)
				.configure(grid0, grid1, grid2, block0, block1, block2).launch(deviceName, args);
	}

	/**
	 * Launch a CUDA kernel with dynamic shared memory.
	 * 
	 * @param kernelString string containing the CUDA kernelcode
	 * @param kernelName   name of the kernel
	 * @param options      options for the nvtrc compiler
	 * @param grid0        number of grids for kernellaunch in first dimension
	 * @param grid1        number of grids for kernellaunch in second dimension
	 * @param grid2        number of grids for kernellaunch in third dimension
	 * @param block0       number of blocks for kernellaunch in first dimension
	 * @param block1       number of blocks for kernellaunch in second dimension
	 * @param block2       number of blocks for kernellaunch in third dimension
	 * @param shared       amount of dynamic shared memory in bytes
	 * @param args         KernelArgs
	 * @return KernelTime for the Execution of this Kernel
	 */
	public static KernelTime launch(String kernelString, String kernelName, Options options, int grid0, int grid1,
			int grid2, int block0, int block1, int block2, long shared, KernelArg... args) {
		return Program.create(kernelString, kernelName).compile(options)
				.configure(grid0, grid1, grid2, block0, block1, block2, shared).launch(args);
	}

	/**
	 * Launch a CUDA kernel with dynamic shared memory.
	 * 
	 * @param kernelString string containing the CUDA kernelcode
	 * @param kernelName   name of the kernel
	 * @param options      options for the nvtrc compiler
	 * @param deviceName   name of the device on which the kernel should be launched
	 * @param grid0        number of grids for kernellaunch in first dimension
	 * @param grid1        number of grids for kernellaunch in second dimension
	 * @param grid2        number of grids for kernellaunch in third dimension
	 * @param block0       number of blocks for kernellaunch in first dimension
	 * @param block1       number of blocks for kernellaunch in second dimension
	 * @param block2       number of blocks for kernellaunch in third dimension
	 * @param shared       amount of dynamic shared memory in bytes
	 * @param args         KernelArgs
	 * @return KernelTime for the Execution of this Kernel
	 */
	public static KernelTime launch(String kernelString, String kernelName, Options options, String deviceName,
			int grid0, int grid1, int grid2, int block0, int block1, int block2, long shared, KernelArg... args) {
		return Program.create(kernelString, kernelName).compile(options)
				.configure(grid0, grid1, grid2, block0, block1, block2, shared).launch(deviceName, args);
	}

	/**
	 * Launch a CUDA kernel with template parameters.
	 * 
	 * @param kernelString      string containing the CUDA kernelcode
	 * @param kernelName        name of the kernel
	 * @param options           options for the nvtrc compiler
	 * @param deviceName        name of the device on which the kernel should be
	 *                          launched
	 * @param templateParameter array of templateParameters which can not be empty
	 * @param grid0             number of grids for kernellaunch in first dimension
	 * @param grid1             number of grids for kernellaunch in second dimension
	 * @param grid2             number of grids for kernellaunch in third dimension
	 * @param block0            number of blocks for kernellaunch in first dimension
	 * @param block1            number of blocks for kernellaunch in second
	 *                          dimension
	 * @param block2            number of blocks for kernellaunch in third dimension
	 * @param shared            amount of dynamic shared memory in bytes
	 * @param args              KernelArgs
	 * @return KernelTime for the Execution of this Kernel
	 */
	public static KernelTime launch(String kernelString, String kernelName, Options options, String deviceName,
			String[] templateParameter, int grid0, int grid1, int grid2, int block0, int block1, int block2,
			long shared, KernelArg... args) {
		return Program.create(kernelString, kernelName).instantiate(templateParameter).compile(options)
				.configure(grid0, grid1, grid2, block0, block1, block2, shared).launch(deviceName, args);
	}

	/**
	 * Benchmark a CUDA kernel loading the kernel string from a file in directory
	 * "kernels" with kernelname.cu as filename.
	 * 
	 * @param kernelName       name of the kernel
	 * @param options          options for the nvtrc compiler
	 * @param numberExecutions number of executions for the kernel
	 * @param creator          KernelArgCreator for creating KernelArgs for the
	 *                         kernel
	 * @param dataSizesBytes   data sizes of the kernel arguments in bytes
	 * @return result of benchmark-test
	 */
	public static BenchmarkResult benchmark(String kernelName, Options options, int numberExecutions,
			KernelArgCreator creator, long... dataSizesBytes) throws IOException {
		return benchmark(Utils.loadFile("kernels/" + kernelName + ".cu"), kernelName, options, Devices.findDevice(),
				numberExecutions, creator, dataSizesBytes);
	}

	/**
	 * Benchmark a CUDA kernel.
	 * 
	 * @param kernelString     string containing the CUDA kernelcode
	 * @param kernelName       name of the kernel
	 * @param options          options for the nvtrc compiler
	 * @param device           device on which the benchmark-test should be executed
	 * @param numberExecutions number of executions for the kernel
	 * @param creator          KernelArgCreator for creating KernelArgs for the
	 *                         kernel for every data size
	 * @param dataSizesBytes   data sizes of the kernel arguments, which should be
	 *                         tested, in bytes
	 * @return result of benchmark-test
	 */
	public static BenchmarkResult benchmark(String kernelString, String kernelName, Options options, Device device,
			int numberExecutions, KernelArgCreator creator, long... dataSizesBytes) {
		return benchmark(kernelString, kernelName, options, device, new String[0], numberExecutions, creator,
				dataSizesBytes);
	}

	/**
	 * Benchmark a CUDA kernel.
	 * 
	 * @param kernelString      string containing the CUDA kernelcode
	 * @param kernelName        name of the kernel
	 * @param options           options for the nvtrc compiler
	 * @param device            device on which the benchmark-test should be
	 *                          executed
	 * @param templateParameter array of templateParameters or an empty array if the
	 *                          kernel do not contains template parameters
	 * @param numberExecutions  number of executions for the kernel for every data
	 *                          size
	 * @param creator           KernelArgCreator for creating KernelArgs for the
	 *                          kernel
	 * @param dataSizesBytes    data sizes of the kernel arguments, which should be
	 *                          tested, in bytes
	 * @return result of benchmark-test
	 */
	public static BenchmarkResult benchmark(String kernelString, String kernelName, Options options, Device device,
			String[] templateParameter, int numberExecutions, KernelArgCreator creator, long... dataSizesBytes) {
		if (dataSizesBytes == null)
			throw new NullPointerException();
		if (dataSizesBytes.length == 0)
			throw new IllegalArgumentException("not data sizes specificated");
		if (numberExecutions <= 0)
			throw new IllegalArgumentException("illegal number of executions: " + numberExecutions);

		// Absolute time Measurement
		long t0 = System.currentTimeMillis();

		// Create and compile Kernel
		Program program = Program.create(kernelString, kernelName);
		if (templateParameter.length > 0)
			program.instantiate(templateParameter);
		Kernel kernel = program.compile(options);

		// Array for result
		KernelTime[][] result = new KernelTime[dataSizesBytes.length][numberExecutions];

		// Start run for every dataSize
		for (int i = 0; i < dataSizesBytes.length; i++) {
			long dataSize = dataSizesBytes[i];

			if (dataSize <= 0)
				throw new IllegalArgumentException();

			// Configure Kernel
			int dataLength = creator.getDataLength(dataSize);
			kernel.configure(creator.getGrid0(dataLength), creator.getGrid1(dataLength), creator.getGrid2(dataLength),
					creator.getBlock0(dataLength), creator.getBlock1(dataLength), creator.getBlock2(dataLength),
					creator.getSharedMemory(dataSize));

			// Create KernelArgs for this dataSize
			KernelArg[] args = creator.createArgs(dataLength);

			// Execute Kernel numberExecutions times
			result[i] = benchmark(kernel, device, args, numberExecutions);

			// Destroy corresponding C++-Objects
			for (KernelArg arg : args) {
				arg.dispose();
			}
		}

		// Absolute time Measurement
		long dt = System.currentTimeMillis() - t0;

		return new BenchmarkResult(device, numberExecutions, dataSizesBytes, result, kernelName, dt);
	}

	/**
	 * Execute a kernel <code>numberExecutions</code> times.
	 * 
	 * @param kernel           kernel, which should be executed
	 * @param device           device on which the kernel should be launched
	 * @param args             KernelArgs
	 * @param numberExecutions number of executions of the kernel
	 * @return KernelTimes for the Execution of this Kernel
	 */
	private static native KernelTime[] benchmark(Kernel kernel, Device device, KernelArg[] args, int numberExecutions);

	/**
	 * Abstract class for generate KernelArgs with a specific size.
	 */
	public static abstract class KernelArgCreator {
		/**
		 * Returns the length of the data (number of elements).
		 * 
		 * @param dataSizeBytes size of data in bytes
		 * @return length of the data
		 */
		public abstract int getDataLength(long dataSizeBytes);

		/**
		 * Generate KernelArgs.
		 * 
		 * @param dataLength length of the data (number of elements)
		 * @return KernelArgs
		 */
		public abstract KernelArg[] createArgs(int dataLength);

		/**
		 * Returns the number of grids for kernellaunch in first dimension.
		 * 
		 * @param dataLength length of the data (number of elements)
		 * @return number of grids for kernellaunch in first dimension
		 */
		public abstract int getGrid0(int dataLength);

		/**
		 * Returns the number of grids for kernellaunch in second dimension.
		 * 
		 * @param dataLength length of the data (number of elements)
		 * @return number of grids for kernellaunch in second dimension
		 */
		public int getGrid1(int dataLength) {
			return 1;
		}

		/**
		 * Returns the number of grids for kernellaunch in third dimension.
		 * 
		 * @param dataLength length of the data (number of elements)
		 * @return number of grids for kernellaunch in third dimension
		 */
		public int getGrid2(int dataLength) {
			return 1;
		}

		/**
		 * Returns the number of blocks for kernellaunch in first dimension.
		 * 
		 * @param dataLength length of the data (number of elements)
		 * @return number of blocks for kernellaunch in first dimension
		 */
		public abstract int getBlock0(int dataLength);

		/**
		 * Returns the number of blocks for kernellaunch in second dimension.
		 * 
		 * @param dataLength length of the data (number of elements)
		 * @return number of blocks for kernellaunch in second dimension
		 */
		public int getBlock1(int dataLength) {
			return 1;
		}

		/**
		 * Returns the number of blocks for kernellaunch in third dimension.
		 * 
		 * @param dataLength length of the data (number of elements)
		 * @return number of blocks for kernellaunch in third dimension
		 */
		public int getBlock2(int dataLength) {
			return 1;
		}

		/**
		 * Returns the amount of dynamic shared memory in bytes.
		 * 
		 * @param dataSizeBytes dataSizeBytes size of data in bytes
		 * @return dynamic shared memory in bytes
		 */
		public long getSharedMemory(long dataSizeBytes) {
			return 0;
		}
	}

	/**
	 * Class representing the result of a benchmark-test.
	 */
	public static class BenchmarkResult {
		private final String deviceInformation;
		private final int numberExecutions;
		private final long[] dataSizes;
		private final KernelTime[][] result;
		private final KernelTime[] average;
		private final String kernelName;
		private final long testDuration;

		/**
		 * Create a new result of benchmark-test.
		 * 
		 * @param device       device on which the benchmark-test was executed
		 * @param executions   number of executions for the kernel for every data size
		 * @param dataSizes    data sizes of the kernel arguments, which was tested, in
		 *                     bytes
		 * @param result       KernelTimes for every kernel execution for every datasize
		 * @param kernelName   name of the tested kernel
		 * @param testDuration duration of the test in milliseconds
		 */
		protected BenchmarkResult(Device device, int numberExecutions, long[] dataSizes, KernelTime[][] result,
				String kernelName, long testDuration) {
			this.numberExecutions = numberExecutions;
			this.dataSizes = dataSizes;
			Arrays.parallelSort(dataSizes);
			this.result = result;
			this.kernelName = kernelName;
			this.testDuration = testDuration;

			deviceInformation = "Device: " + device.getName();

			// Compute Average
			average = new KernelTime[result.length];
			for (int i = 0; i < dataSizes.length; i++) {
				double upload = 0;
				double download = 0;
				double launch = 0;
				double total = 0;

				for (int j = 0; j < numberExecutions; j++) {
					upload += result[i][j].getUpload();
					download += result[i][j].getDownload();
					launch += result[i][j].getLaunch();
					total += result[i][j].getTotal();
				}

				average[i] = new KernelTime((float) (upload / numberExecutions), (float) (download / numberExecutions),
						(float) (launch / numberExecutions), (float) (total / numberExecutions));
			}
		}

		/**
		 * Create a new result of benchmark-test.
		 * 
		 * @param deviceInformation String with deviceInformation
		 * @param numberExecutions  number of executions for the kernel for every data
		 *                          size
		 * @param dataSizes         data sizes of the kernel arguments, which was
		 *                          tested, in bytes
		 * @param result            KernelTimes for every kernel execution for every
		 *                          datasize
		 * @param average           average of result
		 * @param kernelName        name of the tested kernel
		 * @param testDuration      duration of the test in milliseconds
		 */
		private BenchmarkResult(String deviceInformation, int numberExecutions, long[] dataSizes, KernelTime[][] result,
				KernelTime[] average, String kernelName, long testDuration) {
			this.deviceInformation = deviceInformation;
			this.numberExecutions = numberExecutions;
			this.dataSizes = dataSizes;
			this.result = result;
			this.average = average;
			this.kernelName = kernelName;
			this.testDuration = testDuration;
		}

		/**
		 * Returns the number of executions for the kernel for every data size.
		 * 
		 * @return number of executions
		 */
		public int getNumberExecutions() {
			return numberExecutions;
		}

		/**
		 * Returns the data sizes of the kernel arguments, which was tested, in bytes.
		 * 
		 * @return data sizes, which was tested
		 */
		public long[] getDataSizes() {
			return dataSizes;
		}

		/**
		 * Returns the KernelTimes for every kernel execution for every datasize.
		 * 
		 * @return KernelTimes for kernel executions
		 */
		public KernelTime[][] getResult() {
			return result;
		}

		/**
		 * Returns the average KernelTimes for one kernel execution for every datasize.
		 * 
		 * @return average KernelTimes for one kernel execution for every datasize
		 */
		public KernelTime[] getAverage() {
			return average;
		}

		/**
		 * Returns the name of the tested kernel.
		 * 
		 * @return name of the tested kernel
		 */
		public String getKernelName() {
			return kernelName;
		}

		/**
		 * Adds the result of this Benchmark to another BenchmarkResult.
		 * 
		 * @param benchmark BenchmarkResult, which should be added
		 * @return sum of the benchmarks
		 */
		public BenchmarkResult addBenchmarkResult(BenchmarkResult benchmark) {
			if (numberExecutions != benchmark.numberExecutions)
				throw new IllegalArgumentException("Both benchmark result must have the same number of executions");
			for (int i = 0; i < dataSizes.length; i++)
				if (dataSizes[i] != benchmark.dataSizes[i])
					throw new IllegalArgumentException("Both benchmark result must have the same dataSizes");

			String deviceInformation;
			if (this.deviceInformation.equals(benchmark.deviceInformation))
				deviceInformation = this.deviceInformation;
			else
				deviceInformation = this.deviceInformation + " and " + benchmark.deviceInformation;

			String kernelName = this.kernelName + " and " + benchmark.kernelName;
			long testDuration = this.testDuration + benchmark.testDuration;

			KernelTime[][] result = new KernelTime[dataSizes.length][numberExecutions];
			KernelTime[] average = new KernelTime[dataSizes.length];

			for (int i = 0; i < dataSizes.length; i++) {
				for (int j = 0; j < numberExecutions; j++) {
					result[i][j] = this.result[i][j].addKernelTime(benchmark.result[i][j]);
				}

				average[i] = this.average[i].addKernelTime(benchmark.average[i]);
			}

			return new BenchmarkResult(deviceInformation, this.numberExecutions, this.dataSizes, result, average,
					kernelName, testDuration);
		}

		@Override
		public String toString() {
			StringBuffer buffer = new StringBuffer(200);
			buffer.append("\nBenchmark " + kernelName + "-Kernel");
			buffer.append(deviceInformation + "\n");

			buffer.append("  Datasize  Result (Average)\n");

			// For every dataSize: Append average for one kernel execution
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

			// Absolute execution-time of the test
			DecimalFormat df = new DecimalFormat();
			String time = KernelTime.humanReadableMilliseconds(df, testDuration);
			df.setMaximumFractionDigits(1);
			String[] s = time.split(" ");

			if (s.length == 3)
				buffer.append("\nBenchmark-Duration: " + df.format(Double.parseDouble(s[0])) + " " + s[2] + "\n");
			else
				buffer.append("\nBenchmark-Duration: " + df.format(Double.parseDouble(s[0])) + " " + s[1] + "\n");

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
