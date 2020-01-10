package yacx;

/**
 * Class to help launch and configure a CUDA kernel.
 */
public class Kernel extends JNIHandle {
    private boolean configured;

    /**
     * Configure the kernel.
     * @param grid0 number of grids in first dimension
     * @param grid1 number of grids in second dimension
     * @param grid2 number of grids in third dimension
     * @param block0 number of blocks in first dimension
     * @param block1 number of blocks in second dimension
     * @param block2 number of blocks in third dimension
     * @return <code>this</code>
     */
    public Kernel configure(int grid0, int grid1, int grid2, int block0, int block1, int block2) {
        assert(grid0 > 0 && grid1 > 0 && grid2 > 0);
        assert(block0 > 0 && block1 > 0 && block2 > 0);

        configureInternal(grid0, grid1, grid2, block0, block1, block2);

        configured = true;

        return this;
    }

    /**
     * Configure the kernel.
     * @param grid number of grids
     * @param block number of blocks
     * @return <code>this</code>
     */
    public Kernel configure(int grid, int block) {
        assert(grid > 0 && block > 0);

        configureInternal(grid, 1, 1, block, 1, 1);

        configured = true;

        return this;
    }

    private native void configureInternal(int grid0, int grid1, int grid2, int block0, int block1, int block2);

    /**
     * Launch the kernel. <br>
     * Before the kernel can be launched the kernel must be configured.
     * @param args KernelArgs
     * @return KernelTime for the Execution of this Kernel
     */
    public KernelTime launch(KernelArg ...args) {
        assert(args != null && args.length > 0);
        if (!configured)
            throw new IllegalStateException("Kernel not configured");

        return launchInternal(args);
    }

    /**
     * Configure and launch the kernel.
     * @param grid number of grids
     * @param block number of blocks
     * @param args KernelArgs
     * @return KernelTime for the Execution of this Kernel
     */
    public KernelTime launch(int grid, int block, KernelArg ...args) {
        configure(grid, block);
        
        return launch(args);
    }

    /**
     * Configure and launch the kernel.
     * @param grid0 number of grids in first dimension
     * @param grid1 number of grids in second dimension
     * @param grid2 number of grids in third dimension
     * @param block0 number of blocks in first dimension
     * @param block1 number of blocks in second dimension
     * @param block2 number of blocks in third dimension
     * @param args KernelArgs
     * @return KernelTime for the Execution of this Kernel
     */
    public KernelTime launch(int grid0, int grid1, int grid2, int block0, int block1, int block2, KernelArg ...args) {
        configure(grid0, grid1, grid2, block0, block1, block2);
        
        return launch(args);
    }

    private native KernelTime launchInternal(KernelArg[] args);

    /**
     * Launch the kernel. <br>
     * Before the kernel can be launched the kernel must be configured.
     * @param device device on which the kernel should be launched
     * @param args KernelArgs
     * @return KernelTime for the Execution of this Kernel
     */
    public KernelTime launch(Device device, KernelArg ...args) {
        assert(args != null && args.length > 0);
        assert(device != null);
        if (!configured)
            throw new IllegalStateException("Kernel not configured");

        return launchInternal(device, args);
    }

    private native KernelTime launchInternal(Device device, KernelArg[] args);

    /**
     * Launch the kernel. <br>
     * Before the kernel can be launched the kernel must be configured.
     * @param name of the  device device on which the kernel should be launched
     * @param args KernelArgs
     * @return KernelTime for the Execution of this Kernel
     */
    public KernelTime launch(String devicename, KernelArg ...args) {
        assert(args != null && args.length > 0);
        assert(devicename != null && devicename.length() > 0);
        if (!configured)
            throw new IllegalStateException("Kernel not configured");

        return launchInternal(devicename, args);
    }

    private native KernelTime launchInternal(String devicename, KernelArg[] args);

    /**
	 * Create a new Program.
	 * @param handle Pointer to corresponding C-Kernel-Object
	 */
    Kernel(long handle) {
        super(handle);

        configured = false;
    }
}
