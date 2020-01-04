import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestMethodOrder;
import org.junit.jupiter.api.MethodOrderer.OrderAnnotation;

@TestMethodOrder(OrderAnnotation.class)
class TestKernel extends TestJNI {
	final static String archOption = "--gpu-architecture";
	final static String archValue = "compute_35";
	
	static Kernel kernelDims;
	
	static IntArg grid0, grid1, grid2, block0, block1, block2;
	
	@BeforeAll
	static void init() {
		//Kernel for get blockDims and gridDims from running kernel
		String kernelDimsString = "extern \"C\" __global__\n" + 
				"void kernelDims(int* gridx, int* gridy, int* gridz, int* blockx, int* blocky, int* blockz) {\n" + 
				"  *blockx = blockDim.x;\n" + 
				"  *blocky = blockDim.y;\n" + 
				"  *blockz = blockDim.z;\n" + 
				"  *gridx = gridDim.x;\n" + 
				"  *gridy = gridDim.y;\n" + 
				"  *gridz = gridDim.z;\n" + 
				"}\n" + 
				"";
		
		//Create a Option (not necessary)
		Options options = Options.createOptions();
		options.insert(archOption, archValue);
		
		kernelDims = Program.create(kernelDimsString, "kernelDims").compile(options);
		
		grid0 = IntArg.createOutput(1);
		grid1 = IntArg.createOutput(1);
		grid2 = IntArg.createOutput(1);
		block0 = IntArg.createOutput(1);
		block1 = IntArg.createOutput(1);
		block2 = IntArg.createOutput(1);
	}
	
	/**
	 * Checks if the kernel runs with the expected number of grids and blocks
	 */
	void checkDims(int grid0, int grid1, int grid2, int block0, int block1, int block2) {
		kernelDims.launch(TestKernel.grid0, TestKernel.grid1, TestKernel.grid2,
							TestKernel.block0, TestKernel.block1, TestKernel.block2);
		
		assertEquals(grid0, TestKernel.grid0.asIntArray()[0]);
		assertEquals(grid1, TestKernel.grid1.asIntArray()[0]);
		assertEquals(grid2, TestKernel.grid2.asIntArray()[0]);
		assertEquals(block0, TestKernel.block0.asIntArray()[0]);
		assertEquals(block1, TestKernel.block1.asIntArray()[0]);
		assertEquals(block2, TestKernel.block2.asIntArray()[0]);
	}
	
	@Test
	void testConfigureInvalid() {
		//Test configuring Kernel with invalid parameters for grid/block
		assertThrows(IllegalArgumentException.class, () -> {
			kernelDims.configure(0, 1);
		});
		
		assertThrows(IllegalArgumentException.class, () -> {
			kernelDims.configure(1, 0);
		});
		
		assertThrows(IllegalArgumentException.class, () -> {
			kernelDims.configure(0, 0, 0, 0, 0, 0);
		});
		
		assertThrows(IllegalArgumentException.class, () -> {
			kernelDims.configure(-1, 3, 4, 5, 6, 7);
		});
		
		assertThrows(IllegalArgumentException.class, () -> {
			kernelDims.configure(4, 4, 4, 4, 4, 0);
		});
		
		assertThrows(IllegalArgumentException.class, () -> {
			kernelDims.configure(Integer.MIN_VALUE, 1);
		});
	}

	@Test
	void testConfigureValid() {
		//Configuring kernel with different number of threads and blocks and check if kernel run
		//with this expected number of threads and blocks
		kernelDims.configure(1, 1);
		checkDims(1, 1, 1, 1, 1, 1);
		
		kernelDims.configure(4, 5);
		checkDims(4, 1, 1, 5, 1, 1);
		
		kernelDims.configure(9, 8, 7, 6, 5, 4);
		checkDims(9, 8, 7, 6, 5, 4);
		
		kernelDims.configure(15, 7, 8, 29, 17, 1);
		checkDims(15, 7, 8, 29, 17, 1);
	}
	
	@Test
	@Order(1) //Should be first test otherwise it doesnt work
	void testLaunchInvalidNotConfigured() {
		//Run with a not configured Kernel
		assertThrows(IllegalStateException.class, () -> {
			kernelDims.launch(grid0, grid1, grid2, block0, block1, block2);
		});
	}
	
	@Test
	@Order(2) //Should be second test otherwise it doesnt work
	void testLaunchInvalid() {
		//Try to run after invalid configuration
		assertThrows(IllegalArgumentException.class, () -> {
			kernelDims.configure(1, 0);
		});
		
		assertThrows(IllegalStateException.class, () -> {
			kernelDims.launch(grid0, grid1, grid2, block0, block1, block2);
		});
		
		//Configure Kernel
		kernelDims.configure(4, 5, 1, 3, 6, 2);
		
		//Check if parameter null
		assertThrows(NullPointerException.class, () -> {
			kernelDims.launch((KernelArg[]) null);
		});
		
		assertThrows(NullPointerException.class, () -> {
			kernelDims.launch((KernelArg) null);
		});
		
		assertThrows(NullPointerException.class, () -> {
			kernelDims.launch((KernelArg) null, grid1, grid2, block0, block1, block2);
		});
		
		assertThrows(NullPointerException.class, () -> {
			kernelDims.launch(grid0, grid1, grid2, block0, null, block2);
		});
		
		//Check launch without parameters
		assertThrows(IllegalArgumentException.class, () -> {
			kernelDims.launch();
		});
	}
	
	@Test
	void testLaunch() {
		//Configure Kernel
		kernelDims.configure(4, 5, 1, 3, 6, 2);
		
		//Launch Kernel correctly
		kernelDims.launch(grid0, grid1, grid2, block0, block1, block2);
		
		//Check Result
		assertEquals(1, grid0.asIntArray().length);
		assertEquals(1, grid1.asIntArray().length);
		assertEquals(1, grid2.asIntArray().length);
		assertEquals(1, block0.asIntArray().length);
		assertEquals(1, block1.asIntArray().length);
		assertEquals(1, block2.asIntArray().length);
		
		assertEquals(4, grid0.asIntArray()[0]);
		assertEquals(5, grid1.asIntArray()[0]);
		assertEquals(1, grid2.asIntArray()[0]);
		assertEquals(3, block0.asIntArray()[0]);
		assertEquals(6, block1.asIntArray()[0]);
		assertEquals(2, block2.asIntArray()[0]);
		
		//launch Kernel correctly again with other configuration and device-parameter
		kernelDims.configure(42, 1, 3, 19, 7, 7);
		Device device = Device.createDevice();
		
		kernelDims.launch(device, grid0, grid1, grid2, block0, block1, block2);
		
		//Check Result
		assertEquals(1, grid0.asIntArray().length);
		assertEquals(1, grid1.asIntArray().length);
		assertEquals(1, grid2.asIntArray().length);
		assertEquals(1, block0.asIntArray().length);
		assertEquals(1, block1.asIntArray().length);
		assertEquals(1, block2.asIntArray().length);
		
		assertEquals(42, grid0.asIntArray()[0]);
		assertEquals(1, grid1.asIntArray()[0]);
		assertEquals(3, grid2.asIntArray()[0]);
		assertEquals(19, block0.asIntArray()[0]);
		assertEquals(7, block1.asIntArray()[0]);
		assertEquals(7, block2.asIntArray()[0]);
		
		//launch Kernel correctly again with other configuration and devicename-parameter
		kernelDims.configure(17, 4);
		
		kernelDims.launch(device.getName(), grid0, grid1, grid2, block0, block1, block2);
		
		//Check Result
		assertEquals(1, grid0.asIntArray().length);
		assertEquals(1, grid1.asIntArray().length);
		assertEquals(1, grid2.asIntArray().length);
		assertEquals(1, block0.asIntArray().length);
		assertEquals(1, block1.asIntArray().length);
		assertEquals(1, block2.asIntArray().length);
		
		assertEquals(17, grid0.asIntArray()[0]);
		assertEquals(1, grid1.asIntArray()[0]);
		assertEquals(1, grid2.asIntArray()[0]);
		assertEquals(4, block0.asIntArray()[0]);
		assertEquals(1, block1.asIntArray()[0]);
		assertEquals(1, block2.asIntArray()[0]);
	}
	
	@Test
	void testConfigureLaunchInvalid() {
		//Test configuring and launch Kernel with invalid parameters for grid/block
		assertThrows(IllegalArgumentException.class, () -> {
			kernelDims.launch(0, 1, grid0, grid1, grid2, block0, block1, block2);
		});
		
		assertThrows(IllegalArgumentException.class, () -> {
			kernelDims.launch(1, 0, grid0, grid1, grid2, block0, block1, block2);
		});
		
		assertThrows(IllegalArgumentException.class, () -> {
			kernelDims.launch(0, 0, 0, 0, 0, 0, grid0, grid1, grid2, block0, block1, block2);
		});
		
		assertThrows(IllegalArgumentException.class, () -> {
			kernelDims.launch(-1, 3, 4, 5, 6, 7, grid0, grid1, grid2, block0, block1, block2);
		});
		
		assertThrows(IllegalArgumentException.class, () -> {
			kernelDims.launch(4, 4, 4, 4, 4, 0, grid0, grid1, grid2, block0, block1, block2);
		});
		
		assertThrows(IllegalArgumentException.class, () -> {
			kernelDims.launch(Integer.MIN_VALUE, 1, grid0, grid1, grid2, block0, block1, block2);
		});
	}

	@Test
	void testConfigureLaunchValid() {		
		//Launch correctly configured Kernel
		kernelDims.launch(4, 5, grid0, grid1, grid2, block0, block1, block2);
		
		//Check Result
		assertEquals(1, grid0.asIntArray().length);
		assertEquals(1, grid1.asIntArray().length);
		assertEquals(1, grid2.asIntArray().length);
		assertEquals(1, block0.asIntArray().length);
		assertEquals(1, block1.asIntArray().length);
		assertEquals(1, block2.asIntArray().length);
		
		assertEquals(4, grid0.asIntArray()[0]);
		assertEquals(1, grid1.asIntArray()[0]);
		assertEquals(1, grid2.asIntArray()[0]);
		assertEquals(5, block0.asIntArray()[0]);
		assertEquals(1, block1.asIntArray()[0]);
		assertEquals(1, block2.asIntArray()[0]);
		
		
		//Launch other correctly configured Kernel
		kernelDims.launch(4, 3, 9, 5, 18, 2, new KernelArg[] {grid0, grid1, grid2, block0, block1, block2});
		
		//Check Result
		assertEquals(1, grid0.asIntArray().length);
		assertEquals(1, grid1.asIntArray().length);
		assertEquals(1, grid2.asIntArray().length);
		assertEquals(1, block0.asIntArray().length);
		assertEquals(1, block1.asIntArray().length);
		assertEquals(1, block2.asIntArray().length);
		
		assertEquals(4, grid0.asIntArray()[0]);
		assertEquals(3, grid1.asIntArray()[0]);
		assertEquals(9, grid2.asIntArray()[0]);
		assertEquals(5, block0.asIntArray()[0]);
		assertEquals(18, block1.asIntArray()[0]);
		assertEquals(2, block2.asIntArray()[0]);
	}
}
