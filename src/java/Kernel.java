
public class Kernel extends JNIHandle {
    public void configure(int grid, int block){
        configure(grid, 1, 1, block, 1, 1);
    }

    public native void configure(int grid1, int grid2, int grid3, int block1, int block2, int block3);
    public native void launch(KernelArg[] args);

    public void launch(KernelArg[] args, int grid, int block){
        configure(grid, block);
        launch(args);
    }

    public void launch(KernelArg[] args, int grid1, int grid2, int grid3, int block1, int block2, int block3){
        configure(grid1, grid2, grid3, block1, block2, block3);
        launch(args);
    }

    Kernel(long handle) {
        super(handle);
    }
}
