
public class Kernel extends JNIHandle {
    public native void compile();
    public native void configure(int numThreads, int numBlocks);
    public native void launch(KernelArg[] args);

    public void compileAndLaunch(KernelArg[] args, int numThreads, int numBlocks){
        compile();
        configure(numThreads, numBlocks);
        launch(args);
    }

    Kernel(long handle) {
        super(handle);
    }
}
