
public class Kernel extends JNIHandle {
    private boolean configured;

    public Kernel configure(int grid1, int grid2, int grid3, int block1, int block2, int block3) {
        assert(grid1 > 0 && grid2 > 0 && grid3 > 0);
        assert(block1 > 0 && block2 > 0 && block3 > 0);
        configured = true;

        configureInternal(grid1, grid2, grid3, block1, block2, block3);
        
        return this;
    }

    public Kernel configure(int grid, int block) {
        assert(grid > 0 && block > 0);
        configured = true;

        configureInternal(grid, 1, 1, block, 1, 1);
        
        return this;
    }

    private native void configureInternal(int grid1, int grid2, int grid3, int block1, int block2, int block3);

    public KernelTime launch(KernelArg[] args) {
        assert(args != null);
        if (!configured)
            throw new ExecutorFailureException("Kernel not configured");

        return launchInternel(args);
    }

    public KernelTime launch(KernelArg[] args, int grid, int block) {
        configure(grid, block);
        return launch(args);
    }

    public KernelTime launch(KernelArg[] args, int grid1, int grid2, int grid3, int block1, int block2, int block3) {
        configure(grid1, grid2, grid3, block1, block2, block3);
        return launch(args);
    }

    public native KernelTime launchInternel(KernelArg[] args);

    public KernelTime launch(KernelArg[] args, Device device) {
        assert(args != null);
        assert(device != null);
        if (!configured)
            throw new ExecutorFailureException("Kernel not configured");

        return launchInternel(args, device);
    }

    public native KernelTime launchInternel(KernelArg[] args, Device device);

    public KernelTime launch(KernelArg[] args, String devicename) {
        assert(args != null);
        assert(devicename != null && devicename.length() > 0);
        if (!configured)
            throw new ExecutorFailureException("Kernel not configured");

        return launchInternel(args, devicename);
    }

    public native KernelTime launchInternel(KernelArg[] args, String devicename);

    Kernel(long handle) {
        super(handle);

        configured = false;
    }
}
