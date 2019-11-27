
public class Kernel extends JNIHandle {
    public native void configure(int numThreads, int numBlocks);
    public native void launch(KernelArg[] args);

    public void launch(KernelArg[] args, int numThreads, int numBlocks){
        configure(numThreads, numBlocks);
        launch(args);
    }

    Kernel(long handle) {
        super(handle);
    }
}
