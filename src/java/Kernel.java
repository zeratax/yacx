
public class Kernel extends JNIHandle {
    private boolean configured;

    public void configure(int grid1, int grid2, int grid3, int block1, int block2, int block3) {
        assert(grid1 > 0 && grid2 > 0 && grid3 > 0);
        assert(block1 > 0 && block2 > 0 && block3 > 0);
        configured = true;

        configureInternal(grid1, grid2, grid3, block1, block2, block3);
    }

    public void configure(int grid, int block) {
        assert(grid > 0 && block > 0);
        configured = true;

        configureInternal(grid, 1, 1, block, 1, 1);
    }

    private native void configureInternal(int grid1, int grid2, int grid3, int block1, int block2, int block3);

    public void launch(KernelArg[] args) {
        assert(args != null);
        if (!configured)
            throw new ExecutorFailureException("Kernel not configured");

        launchInternel(args);
    }

    public void launch(KernelArg[] args, int grid, int block) {
        configure(grid, block);
        launch(args);
    }

    public void launch(KernelArg[] args, int grid1, int grid2, int grid3, int block1, int block2, int block3) {
        configure(grid1, grid2, grid3, block1, block2, block3);
        launch(args);
    }

    public native void launchInternel(KernelArg[] args);

    Kernel(long handle) {
        super(handle);

        configured = false;
    }
}
