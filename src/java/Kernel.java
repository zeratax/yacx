
public class Kernel extends JNIHandle {
    private boolean configured;

    public Kernel configure(int grid0, int grid1, int grid2, int block0, int block1, int block2) {
        assert(grid0 > 0 && grid1 > 0 && grid2 > 0);
        assert(block0 > 0 && block1 > 0 && block2 > 0);

        configureInternal(grid0, grid1, grid2, block0, block1, block2);
        
        configured = true;
        
        return this;
    }

    public Kernel configure(int grid, int block) {
        assert(grid > 0 && block > 0);

        configureInternal(grid, 1, 1, block, 1, 1);
        
        configured = true;
        
        return this;
    }

    private native void configureInternal(int grid0, int grid1, int grid2, int block0, int block1, int block2);

    public KernelTime launch(KernelArg ...args) {
        assert(args != null && args.length > 0);
        if (!configured)
            throw new IllegalStateException("Kernel not configured");

        return launchInternal(args);
    }

    public KernelTime launch(int grid, int block, KernelArg ...args) {
        configure(grid, block);
        return launch(args);
    }

    public KernelTime launch(int grid0, int grid1, int grid2, int block0, int block1, int block2, KernelArg ...args) {
        configure(grid0, grid1, grid2, block0, block1, block2);
        return launch(args);
    }

    private native KernelTime launchInternal(KernelArg[] args);

    public KernelTime launch(Device device, KernelArg ...args) {
        assert(args != null && args.length > 0);
        assert(device != null);
        if (!configured)
            throw new IllegalStateException("Kernel not configured");

        return launchInternal(device, args);
    }

    private native KernelTime launchInternal(Device device, KernelArg[] args);

    public KernelTime launch(String devicename, KernelArg ...args) {
        assert(args != null && args.length > 0);
        assert(devicename != null && devicename.length() > 0);
        if (!configured)
            throw new IllegalStateException("Kernel not configured");

        return launchInternal(devicename, args);
    }

    private native KernelTime launchInternal(String devicename, KernelArg[] args);

    Kernel(long handle) {
        super(handle);

        configured = false;
    }
}
